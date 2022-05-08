import bpy
import math
import mathutils
from mathutils import Vector, Matrix
import copy
import numpy
import json


def Get_ImageDims_from_JSON(WorkingData, CameraID):
    #gets image dimensions from the OpenMVG/MVS JSON file
    
    
    
    #default location
    CamDetailFile_location= WorkingData.UnstructuredDirectory +WorkingData.OpenSourceSFM_CamDataFile
    
    #if JSON file is found for previous processes, switch from default input folder for manual loading of input files to folder
    #specified from JSON report
    if WorkingData.JSON_SFM_details is not None:
         print("SFM report JSON supersedence, switching default input folder from " + WorkingData.UnstructuredDirectory + " to " +  WorkingData.JSON_SFM_details["OutputFolder"])
         WorkingData.UnstructuredDirectory=WorkingData.JSON_SFM_details["OutputFolder"] + "/"
         CamDetailFile_location= WorkingData.UnstructuredDirectory +WorkingData.OpenSourceSFM_CamDataFile
         
    try:
        #open json file
        #TODO inefficient way of getting image dimensions - if speed is needed this could be optimised
        with open(CamDetailFile_location) as json_file:
            sfm_Data_file=json.load(json_file)
            #depending on organisation of json file, get all "views" first
            #names expected to be image file with extension 
            for indexer, viewitem in enumerate(sfm_Data_file['views']):
                #get image filename/title
                Temp_Filename=viewitem['value']['ptr_wrapper']['data']['filename']
                if Temp_Filename.lower()==CameraID.lower():
                    #get the associated IDs to look-up details in another section of the json
                    Temp_IdPose=viewitem['value']['ptr_wrapper']['data']['id_pose']
                    IdView=viewitem['value']['ptr_wrapper']['data']['id_view']
                    Temp_Cam_focal_width=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['width']
                    Temp_Cam_focal_height=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['height']
                    return(Temp_Cam_focal_width,Temp_Cam_focal_height)
    except Exception as e:
        print( CamDetailFile_location)
        raise Exception ("Get_ImageDims_from_JSON: Could not find image dimensions in JSON or error reading JSON ", CameraID)
        
                   

def Load_ExternalCameras_if_exist(WorkingData):
    #if using open-source SFM no cameras will be imported - need to generate ourselves from the associated json file
    CamDetailFile_location= WorkingData.UnstructuredDirectory +WorkingData.OpenSourceSFM_CamDataFile
    try:
        #open json file
        with open(CamDetailFile_location) as json_file:
            sfm_Data_file=json.load(json_file)
            #depending on organisation of json file, get all "views" first
            #names expected to be image file with extension 
            for indexer, viewitem in enumerate(sfm_Data_file['views']):
                #get image filename/title
                Temp_Filename=viewitem['value']['ptr_wrapper']['data']['filename']
                #chop out ".extension"
                #Temp_Filename=Temp_Filename[0:(Temp_Filename.find('.'))]
                #get the associated IDs to look-up details in another section of the json
                Temp_IdPose=viewitem['value']['ptr_wrapper']['data']['id_pose']
                IdView=viewitem['value']['ptr_wrapper']['data']['id_view']
                #use ID information of image file to look-up associated data 
                Temp_Cam_position=sfm_Data_file['extrinsics'][int(Temp_IdPose)]['value']['center']
                Temp_Cam_rotation=sfm_Data_file['extrinsics'][int(Temp_IdPose)]['value']['rotation']
                
                Temp_Cam_PrinciplePoint=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['principal_point']
                Temp_Cam_focal_length_pixels=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['focal_length']
                Temp_Cam_focal_width=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['width']
                Temp_Cam_focal_height=sfm_Data_file['intrinsics'][int(Temp_IdPose)]['value']["ptr_wrapper"]["data"]['height']
                
                #Blender world matrix is a 4*4 with only first 3*3 populated 
                TemplateMatrix=Matrix()#([0,0,0],[0,0,0],[0,0,0]))
            
                #populate rotation matrix
                TemplateMatrix[0]=mathutils.Vector((Temp_Cam_rotation[0][0], Temp_Cam_rotation[1][0], Temp_Cam_rotation[2][0],0))
                TemplateMatrix[1]=mathutils.Vector((Temp_Cam_rotation[0][1], Temp_Cam_rotation[1][1], Temp_Cam_rotation[2][1],0))
                TemplateMatrix[2]=mathutils.Vector((Temp_Cam_rotation[0][2], Temp_Cam_rotation[1][2], Temp_Cam_rotation[2][2],0))
  
                #apply rotation matrix to object by decomposing into euler angles
                EulerAngles=TemplateMatrix.to_euler('XYZ')
                
                #create camera datablock
                camera_data = bpy.data.cameras.new(name=Temp_Filename)
                camera_object = bpy.data.objects.new('Camera', camera_data)
                camera_object.name=Temp_Filename
                bpy.context.scene.collection.objects.link(camera_object)
                
                #deselect anything active in Blender
                #GeneralFunctions.DeselectAll()        
                
                #create camera object and rename
                CameraRef = bpy.data.objects[Temp_Filename]
                CameraRef.rotation_euler=(EulerAngles[0], EulerAngles[1], EulerAngles[2])
                CameraRef.location=(Temp_Cam_position[0], Temp_Cam_position[1], Temp_Cam_position[2])
                
                #debug - force rotations so we can move on in testing
                CameraRef.data.sensor_height=24
                CameraRef.data.sensor_width=36
                CameraRef.data.sensor_fit='AUTO'
                
                #fix rotation to match the format subsequent operations expect
                CameraRef.rotation_euler.rotate_axis('Y',math.pi)
                CameraRef.rotation_euler.rotate_axis('Z',math.pi)
       
                
        
                
    except Exception as e:
        print("JSON skipped, error or does not exist (valid if camera objects are part of import object) ", e)



def DebugPrint_Landmarks_using_NearestVertex(InputMeshID, Vertex_Vs_FacialLandmarks_dict,GeneralFunctions):
    #parse a dictionary of format key=facial landmark index, value=vertex id
    
    #select generic model
    GeneralFunctions.DeselectAll()
    GeneralFunctions.SelectObject(InputMeshID)

    #get reference to model
    GenericMesh_reference = bpy.data.objects.get(InputMeshID)
    GenericMesh_Num_Vertices=len(GenericMesh_reference.data.vertices.items())
    
    #get local and global vertices 
    Mesh_vertices_local= [GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    Mesh_vertices_global= [GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    
    #set size for debug points/text
    size=0.02
    
    for VertexKey in Vertex_Vs_FacialLandmarks_dict:
        bpy.ops.object.empty_add(radius=size, location=(Mesh_vertices_global[Vertex_Vs_FacialLandmarks_dict[VertexKey]]))
        #print DLib INDEX at landmark location - so we can check the points are legitimate
        bpy.ops.object.text_add(radius=size/2,enter_editmode=False, align='WORLD', location=(Mesh_vertices_global[Vertex_Vs_FacialLandmarks_dict[VertexKey]]),rotation=(90.0, 0.0, 180.0))
        #select last object (text object) and label the landmark with the dlib index
        ob=bpy.context.object
        ob.data.body = str(VertexKey)

def Destructive_RotateDictionaryOfLandmarks(Input_Dictionary, RotationMatrix):
    ###input dictionary of facial landmarks, key=dlib facial index, value = custom class with .OutputPoint being modified

    for Facial_LandMarkKey in Input_Dictionary:
        
        try:
            #get 3d position of facial landmark according to dlib indexing
            Vector2Modify=Input_Dictionary[Facial_LandMarkKey].OutputPoint[0]
            #multiply by rotation matrix
            Vector2Modify=Vector2Modify @ RotationMatrix
            #insert back into dictionary
            Input_Dictionary[Facial_LandMarkKey].OutputPoint[0]= Vector2Modify
        except:
            
            print("Generic Facial Landmark rotate error with facial index " , str(Facial_LandMarkKey))

def GetClosestVertex_of_FacialLandmarks(MeshID, DictionaryOfLandmarks,GeneralFunctions):
    #for each facial landmark, find closest vertex. Build a dictionary of key=vertex ID and value= facial landmark ID
    def GetDistance(P1,P2):
        #euclidian distance between two 3D points
        return math.sqrt(((P2[0]-P1[0])**2) + ((P2[1]-P1[1])**2) + ((P2[2]-P1[2])**2))    
    #make copy of dictionary
    CopyOfLandMark_Dictionary=copy.deepcopy(DictionaryOfLandmarks)
    
    #instantiate new dictionary of vertex positions versus facial landmarks
    Vertex_Vs_Landmark={}
    
    #select generic model
    GeneralFunctions.DeselectAll()
    GeneralFunctions.SelectObject(MeshID)

    #get reference to model
    GenericMesh_reference = bpy.data.objects.get(MeshID)
    GenericMesh_Num_Vertices=len(GenericMesh_reference.data.vertices.items())
    
    #get local and global vertices 
    Mesh_vertices_local= [GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    Mesh_vertices_global= [GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    
    
    #roll through facial landmarks and find nearest vertex - warning - closest vertex may have large error
    for Indexer, facial_index in enumerate(CopyOfLandMark_Dictionary):
       
        #check value is present in BOTH sets of data
        if CopyOfLandMark_Dictionary[facial_index] is None:
            continue#skip to next loop
        if CopyOfLandMark_Dictionary[facial_index].OutputPoint is None:
            continue#skip to next loop

        #the dlib facial landmark at this index are valid for both sets of data
        #generic facial landmark
        GenericTempPoint=[CopyOfLandMark_Dictionary[facial_index].OutputPoint[0][0],CopyOfLandMark_Dictionary[facial_index].OutputPoint[0][1],CopyOfLandMark_Dictionary[facial_index].OutputPoint[0][2]]

        #find closest vertex on generic mesh to current generic facial landmar
        ClosestVertex=None
        SmallestDistance=999999999
        Index_of_vertex=None
        for VertexIndex, Vertex_Position in enumerate(Mesh_vertices_global):
            Distance_Vertex2Landmark=GetDistance(GenericTempPoint,Vertex_Position)
            if Distance_Vertex2Landmark<SmallestDistance:
                SmallestDistance=Distance_Vertex2Landmark
                ClosestVertex=Vertex_Position
                Index_of_vertex=VertexIndex
                
        Vertex_Vs_Landmark[facial_index]=Index_of_vertex
    return Vertex_Vs_Landmark
        

def PreOptimise_Shrinkwrap_ProportionalDeform(GeneralFunctions,GenericMesh_ID, WorkingData,DifferenceDivison=10,BrushSize=5.5,ReverseScan=False):
    #use Blender proportional edit function to stretch generic mesh to conform to unstructured mesh
    #using facial landmark points as targets
    def GetDistance(P1,P2):
        #euclidian distance between two 3D points
        return math.sqrt(((P2[0]-P1[0])**2) + ((P2[1]-P1[1])**2) + ((P2[2]-P1[2])**2))
    
    #select generic model
    GeneralFunctions.DeselectAll()
    GeneralFunctions.SelectObject(WorkingData.GenericHead_name)

    #get reference to model
    GenericMesh_reference = bpy.data.objects.get(GenericMesh_ID)
    GenericMesh_Num_Vertices=len(GenericMesh_reference.data.vertices.items())
    
    #get local and global vertices 
    Mesh_vertices_local= [GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    Mesh_vertices_global= [GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    #Mesh_Vertices_Modified= [GenericMesh_Modified_Reference.matrix_world @ GenericMesh_Modified_Reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    
    #format data
    generic_landmark_positions=[]
    unstructured_landmark_positions=[]
    
    
    #least complex method to reverse dictionary
    ReverseList=[]
    for facial_index in (WorkingData.GenericModel_3D_LandmarksDictionary):
        ReverseList.append(facial_index)
    #calling routine can decide whether to run backwards through sequence of proportional edits    
    if ReverseScan==True:
        ReverseList.reverse()
    
    #run through facial landmark index points of unstructured model to reformat
    for Indexer, facial_index in enumerate(ReverseList):
        #check value is present in BOTH sets of data
        try:
            if WorkingData.GenericModel_3D_LandmarksDictionary[facial_index] is None:
                continue#skip to next loop
            if WorkingData.Unstructured_Model_AverageLandmarksDictionary[facial_index] is None:
                continue #skip to next loop
            if WorkingData.GenericModel_3D_LandmarksDictionary[facial_index].OutputPoint is None:
                continue#skip to next loop
            if WorkingData.Unstructured_Model_AverageLandmarksDictionary[facial_index].OutputPoint is None:
                continue#skip to next loop
            if WorkingData.GenericModel_Vertex_of_FacialLandmark[facial_index] is None:
                continue
        except:
                continue
            
            
        #get generic mesh vertice index previously associated with current facial landmark
        VertexID=WorkingData.GenericModel_Vertex_of_FacialLandmark[facial_index]
        #the dlib facial landmark at this index are valid for both sets of data
        #generic facial landmark (use associated vertex position)
        #get vertex on generic mesh which is closest to facial landmark point - and track during any deformation.. make sure we convert from local space to global
        GenericTempPoint=GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[VertexID].co
        #unstructured facial landmark position in global space
        UnstructuredTempPoint=[WorkingData.Unstructured_Model_AverageLandmarksDictionary[facial_index].OutputPoint[0][0],WorkingData.Unstructured_Model_AverageLandmarksDictionary[facial_index].OutputPoint[0][1],WorkingData.Unstructured_Model_AverageLandmarksDictionary[facial_index].OutputPoint[0][2]]
        #get translation vector between generic facial landmark and unstructured facial landmark
        LandmarkDifference=numpy.subtract(UnstructuredTempPoint,GenericTempPoint)
        #prepare to select vertices
        GeneralFunctions.DeselectAll()
        #select object
        GeneralFunctions.SelectObject(GenericMesh_ID)
        #get reference to object
        obj = bpy.context.active_object
        #set operational mode
        bpy.ops.object.mode_set(mode = 'EDIT') 
        #set select mode
        bpy.ops.mesh.select_mode(type="VERT")
        #necessary step to ensure only one vertex selected
        bpy.ops.mesh.select_all(action = 'DESELECT')
        #set mode again (quirk of blender)
        bpy.ops.object.mode_set(mode = 'OBJECT')
        #select vertex closest to facial landmark
        obj.data.vertices[VertexID].select = True
        #set operational mode
        bpy.ops.object.mode_set(mode = 'EDIT') 
        #perform proportional edit of vertex towards unstructured mesh facial landmark
        #get new landmark position
        NewGenericLandMark=LandmarkDifference/DifferenceDivison
        #print(NewGenericLandMark)
        bpy.ops.transform.translate(value=(NewGenericLandMark), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', use_proportional_edit=True, proportional_edit_falloff='SMOOTH', proportional_size=BrushSize, use_proportional_connected=False, use_proportional_projected=False, release_confirm=True)
        #return to default operational mode
        bpy.ops.object.mode_set(mode = 'OBJECT')





def rigid_transform_3D(A, B):
    
    #code by http://nghiaho.com/?page_id=671
    #by Nghia Ho
    #Reference “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987
    
    # Input: expects 3xN matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix
    # t = 3x1 column vector
    
    #check two sets of data are same size
    assert A.shape == B.shape
    
    #check in right format
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"rigid_transform_3D matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"rigid_transform_3D matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = numpy.mean(A, axis=1)
    centroid_B = numpy.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean to shift sets of data to mean at Origin
    Am = A - centroid_A
    Bm = B - centroid_B
    
    #get covariance matrix
    H = Am @ numpy.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation using singular value decomposition
    U, S, Vt = numpy.linalg.svd(H)
    #formula for finding rotation
    R = Vt.T @ U.T

    # special reflection case (sometimes SVD can return a result which is mirrored(?))
    #checking the determinant can catch this - a valid determinant should be >0
    if numpy.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    #formula to get translation
    t = -R@centroid_A + centroid_B

    return R, t
def RotateAlign_sets_of_3Dpoints(MeshID, DictionaryOfLandmarksA,DictionaryOfLandmarksB):
    #perform Procrustes alignment on a dictionary of landmarks, and apply the transform to the associated mesh
    #two sets of data must be scale-matched first
    
    #format data
    Temp_DLib_Positions_listA=[]
    Temp_DLib_Positions_listB=[]
    #run through dlib points of unstructured model to reformat
    for key in DictionaryOfLandmarksA:
        
        #check value is present in BOTH sets of data
        if DictionaryOfLandmarksA[key] is None:
            continue#skip to next loop
        if DictionaryOfLandmarksB[key] is None:
            continue #skip to next loop
        if DictionaryOfLandmarksA[key].OutputPoint is None:
            continue#skip to next loop
        if DictionaryOfLandmarksB[key].OutputPoint is None:
            continue#skip to next loop
        
        #the dlib facial landmark at this index are valid for both sets of data
        Temp_DLib_Positions_listA.append(DictionaryOfLandmarksA[key].OutputPoint[0][:])#point must be extracted
        Temp_DLib_Positions_listB.append(DictionaryOfLandmarksB[key].OutputPoint[0][:])#point must be extracted
        
    #we now have 2 sets of corresponding points - can now find the transform from one set to another
    #format the lists to a numpy array (matrix) format and transpose to flip the rows/cols
    RotationMatrix, Translation=rigid_transform_3D(numpy.matrix(Temp_DLib_Positions_listA).T,numpy.matrix(Temp_DLib_Positions_listB).T)
    
    #get reference to input mesh
    MeshReference = bpy.data.objects[MeshID] 
    
    #Blender world matrix is a 4*4 with only first 3*3 populated - get object format
    #TemplateMatrix=MeshReference.matrix_world.copy()
    TemplateMatrix=Matrix()#default 4*4 ID matrix 
        
    #populate rotation matrix
    TemplateMatrix[0]=mathutils.Vector((RotationMatrix[0,0], RotationMatrix[0,1], RotationMatrix[0,2],0.0))
    TemplateMatrix[1]=mathutils.Vector((RotationMatrix[1,0], RotationMatrix[1,1], RotationMatrix[1,2],0.0))
    TemplateMatrix[2]=mathutils.Vector((RotationMatrix[2,0], RotationMatrix[2,1], RotationMatrix[2,2],0.0))
    TemplateMatrix[3]=mathutils.Vector((0.0,0.0,0.0,0.0))
    
    #apply rotation matrix to object by decomposing into euler angles and applying offsets
    EulerAngles=TemplateMatrix.to_euler()
    MeshReference.rotation_euler[0]=MeshReference.rotation_euler[0]+EulerAngles[0]
    MeshReference.rotation_euler[1]=MeshReference.rotation_euler[1]+EulerAngles[1]
    MeshReference.rotation_euler[2]=MeshReference.rotation_euler[2]+EulerAngles[2]
    
    return TemplateMatrix, Translation


#return distance of vector
def dist(element1,element2):
    return (element1-element2).length
def ProcrustesAlign(MeshID, DictionaryOfLandmarks,CommonDLIB_Index_ForBasisVector,GeneralFunctions):
    #scale and translate input mesh object using associated facial landmarks
    #return updated facial landmark dictionary
    
    #make copy of dictionary to return the modified version
    CopyOfLandMark_Dictionary=copy.deepcopy(DictionaryOfLandmarks)
    #format data
    Temp_DLib_Positions_list=[]#to hold return from pro
    Temp_DLib_Positions_Indexes=[]
   
    #run through dlib points of unstructured model to reformat, build associated list of indexes
    
    for LandMark in DictionaryOfLandmarks:
        if DictionaryOfLandmarks[LandMark] is None:
            continue#skip this iteration of loop
        if DictionaryOfLandmarks[LandMark].OutputPoint is None:
            continue#skip this iteration of loop
        Temp_DLib_Positions_list.append([DictionaryOfLandmarks[LandMark].OutputPoint[0][0],DictionaryOfLandmarks[LandMark].OutputPoint[0][1],DictionaryOfLandmarks[LandMark].OutputPoint[0][2]])
        Temp_DLib_Positions_Indexes.append(LandMark)

    
    #use arbitrary dlib point for basis vector
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector] is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ", str(CommonDLIB_Index_ForBasisVector))
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector].OutputPoint is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ",str(CommonDLIB_Index_ForBasisVector) )
    
    
    #get procrustes transform - input list of points and an arbitrary point to create basis vector
    LandmarkPositions, transform_x,transform_y,transform_z,Scale_Factor = procrustesComponents(Temp_DLib_Positions_list,DictionaryOfLandmarks[31].OutputPoint[0][:])
    
    #we can assume the facial landmarks and the mesh object are aligned in space - 
    #therefore we can modifiy the mesh object so its origin is the same as the mean of the facial landmarks
    #this will make it easier to modify later on
    
    
    #apply procrustes transform on associated mesh
    #get reference to mesh
    AssociatedMesh = bpy.data.objects.get(MeshID)
    #move cursor location to mean of facial landmark points
    bpy.context.scene.cursor.location = [transform_x,transform_y,transform_z]
    #deselect all objects
    GeneralFunctions.DeselectAll()
    #select input mesh object
    GeneralFunctions.SelectObject(MeshID)
    #set origin to cursor
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    
    
    #set location to origin
    AssociatedMesh.location.x=0
    AssociatedMesh.location.y=0
    AssociatedMesh.location.z=0
    #apply scale
    AssociatedMesh.scale=AssociatedMesh.scale*Scale_Factor
    
    
    #run through the indexes generated above, associated with positions in Temp_DLib_Positions_list
    for ListIndex, LandMarkIndex in enumerate (Temp_DLib_Positions_Indexes):
        #double check the enumerated index matches the landmark index
        #in other words - one list object holds extracted facial landmark INDEXES
        #with each index associated with a position in another list
        #if the enumeration of the key matches then the association is valid.
        #TODO this might have to be refactored using a dictionary from the start
        if str(Temp_DLib_Positions_Indexes[ListIndex])!=str(LandMarkIndex):
            raise Exception("ProcrustesAlign: error for association sanity check")
        
        #with the mean position as the origin of the set - move facial landmarks to origin
        #ListIndex is the zero-based list object index, which can be offset if for example
        #some landmark points had failed and had been omitted from the list
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]-transform_x
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]-transform_y
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]-transform_z
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]*Scale_Factor
        
       
    return CopyOfLandMark_Dictionary


def Scale_Mesh_WithFacialLandmarks_As_Origin(MeshID, DictionaryOfLandmarks,CommonDLIB_Index_ForBasisVector,GeneralFunctions,Scale_Factor):
    #Repurposed function to scale mesh using average point of all facial landmarks as origin
    #scale input mesh object using associated facial landmarks
    #return updated facial landmark dictionary
    
    #make copy of dictionary to return the modified version
    CopyOfLandMark_Dictionary=copy.deepcopy(DictionaryOfLandmarks)
    #format data
    Temp_DLib_Positions_list=[]#to hold return from pro
    Temp_DLib_Positions_Indexes=[]
   
    #run through dlib points of unstructured model to reformat, build associated list of indexes
    
    for LandMark in DictionaryOfLandmarks:
        if DictionaryOfLandmarks[LandMark] is None:
            continue#skip this iteration of loop
        if DictionaryOfLandmarks[LandMark].OutputPoint is None:
            continue#skip this iteration of loop
        Temp_DLib_Positions_list.append([DictionaryOfLandmarks[LandMark].OutputPoint[0][0],DictionaryOfLandmarks[LandMark].OutputPoint[0][1],DictionaryOfLandmarks[LandMark].OutputPoint[0][2]])
        Temp_DLib_Positions_Indexes.append(LandMark)

    
    #use arbitrary dlib point for basis vector
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector] is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ", str(CommonDLIB_Index_ForBasisVector))
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector].OutputPoint is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ",str(CommonDLIB_Index_ForBasisVector) )
    
    
    #get procrustes transform - input list of points and an arbitrary point to create basis vector
    LandmarkPositions, transform_x,transform_y,transform_z,Scale_Factor_unused = procrustesComponents(Temp_DLib_Positions_list,DictionaryOfLandmarks[31].OutputPoint[0][:])
    
    #we can assume the facial landmarks and the mesh object are aligned in space - 
    #therefore we can modifiy the mesh object so its origin is the same as the mean of the facial landmarks
    #this will make it easier to modify later on
    
    
    #apply procrustes transform on associated mesh
    #get reference to mesh
    AssociatedMesh = bpy.data.objects.get(MeshID)
    #move cursor location to mean of facial landmark points
    bpy.context.scene.cursor.location = [transform_x,transform_y,transform_z]
    #deselect all objects
    GeneralFunctions.DeselectAll()
    #select input mesh object
    GeneralFunctions.SelectObject(MeshID)
    #set origin to cursor
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    
    
    #set location to origin
    AssociatedMesh.location.x=0
    AssociatedMesh.location.y=0
    AssociatedMesh.location.z=0
    #apply scale
    AssociatedMesh.scale=AssociatedMesh.scale*Scale_Factor
    
    
    #run through the indexes generated above, associated with positions in Temp_DLib_Positions_list
    for ListIndex, LandMarkIndex in enumerate (Temp_DLib_Positions_Indexes):
        #double check the enumerated index matches the landmark index
        #in other words - one list object holds extracted facial landmark INDEXES
        #with each index associated with a position in another list
        #if the enumeration of the key matches then the association is valid.
        #TODO this might have to be refactored using a dictionary from the start
        if str(Temp_DLib_Positions_Indexes[ListIndex])!=str(LandMarkIndex):
            raise Exception("ProcrustesAlign: error for association sanity check")
        
        #with the mean position as the origin of the set - move facial landmarks to origin
        #ListIndex is the zero-based list object index, which can be offset if for example
        #some landmark points had failed and had been omitted from the list
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]-transform_x
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]-transform_y
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]-transform_z
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]*Scale_Factor
        
       
    return CopyOfLandMark_Dictionary


def ProcrustesAlign_ScaleOnly(MeshID, DictionaryOfLandmarks,CommonDLIB_Index_ForBasisVector,GeneralFunctions):
    #scale input mesh object using associated facial landmarks
    #return updated facial landmark dictionary
    
    #make copy of dictionary to return the modified version
    CopyOfLandMark_Dictionary=copy.deepcopy(DictionaryOfLandmarks)
    #format data
    Temp_DLib_Positions_list=[]#to hold return from pro
    Temp_DLib_Positions_Indexes=[]
   
    #run through dlib points of unstructured model to reformat, build associated list of indexes
    
    for LandMark in DictionaryOfLandmarks:
        if DictionaryOfLandmarks[LandMark] is None:
            continue#skip this iteration of loop
        if DictionaryOfLandmarks[LandMark].OutputPoint is None:
            continue#skip this iteration of loop
        Temp_DLib_Positions_list.append([DictionaryOfLandmarks[LandMark].OutputPoint[0][0],DictionaryOfLandmarks[LandMark].OutputPoint[0][1],DictionaryOfLandmarks[LandMark].OutputPoint[0][2]])
        Temp_DLib_Positions_Indexes.append(LandMark)

    
    #use arbitrary dlib point for basis vector
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector] is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ", str(CommonDLIB_Index_ForBasisVector))
    if DictionaryOfLandmarks[CommonDLIB_Index_ForBasisVector].OutputPoint is None:
        raise Exception("Procrustes Align error: DLIB landmark point used to create basis vector is invalid! ",str(CommonDLIB_Index_ForBasisVector) )
    
    
    #get procrustes transform - input list of points and an arbitrary point to create basis vector
    LandmarkPositions, transform_x,transform_y,transform_z,Scale_Factor = procrustesComponents(Temp_DLib_Positions_list,DictionaryOfLandmarks[31].OutputPoint[0][:])
    
    #we can assume the facial landmarks and the mesh object are aligned in space - 
    #therefore we can modifiy the mesh object so its origin is the same as the mean of the facial landmarks
    #this will make it easier to modify later on
    
    
    #apply procrustes transform on associated mesh
    #get reference to mesh
    AssociatedMesh = bpy.data.objects.get(MeshID)
    #move cursor location to mean of facial landmark points
    bpy.context.scene.cursor.location = [transform_x,transform_y,transform_z]
    #deselect all objects
    GeneralFunctions.DeselectAll()
    #select input mesh object
    GeneralFunctions.SelectObject(MeshID)
    #set origin to cursor
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
    
    
    #set location to origin
    AssociatedMesh.location.x=0
    AssociatedMesh.location.y=0
    AssociatedMesh.location.z=0
    #apply scale
    AssociatedMesh.scale=AssociatedMesh.scale*Scale_Factor
    
    
    #run through the indexes generated above, associated with positions in Temp_DLib_Positions_list
    for ListIndex, LandMarkIndex in enumerate (Temp_DLib_Positions_Indexes):
        #double check the enumerated index matches the landmark index
        #in other words - one list object holds extracted facial landmark INDEXES
        #with each index associated with a position in another list
        #if the enumeration of the key matches then the association is valid.
        #TODO this might have to be refactored using a dictionary from the start
        if str(Temp_DLib_Positions_Indexes[ListIndex])!=str(LandMarkIndex):
            raise Exception("ProcrustesAlign: error for association sanity check")
        
        #with the mean position as the origin of the set - move facial landmarks to origin
        #ListIndex is the zero-based list object index, which can be offset if for example
        #some landmark points had failed and had been omitted from the list
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]-transform_x
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]-transform_y
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]-transform_z
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][0]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][1]*Scale_Factor
        CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]=CopyOfLandMark_Dictionary[LandMarkIndex].OutputPoint[0][2]*Scale_Factor
        
       
    return CopyOfLandMark_Dictionary

def procrustesComponents(points, Common_ArbitraryPoint):

    """Function to return procrustes components for a point set

    Arguments:
    ---------

    points	a list of points in form list.append([x,y,z])
    name	a base name for the locators

    Returns:
    --------

    a list of landmark positions, x, y, z positional offsets and a scale value
    """
    
    k = len(points)
    
    # declare an array to hold the landmark positions
    landmark_positions = []

    # extract the x, y, z, values
    x_vals = [i[0] for i in points]
    y_vals = [i[1] for i in points]
    z_vals = [i[2] for i in points]

    # compute the x, y, z means (centroids)
    x_mean = sum(x_vals) / k
    y_mean = sum(y_vals) / k
    z_mean = sum(z_vals) / k
    
    Point3D=points[0]
    # compute the mean of the squared distances - use any arbitrary point to create a vector - might be worth experimenting using farthest vector for greater accuracy
    #sum_sqr_vals = sum([((Point3D[0] - x_mean) * (Point3D[0] - x_mean)) + ((Point3D[1] - y_mean) * (Point3D[1] - y_mean)) + ((Point3D[2] - z_mean) * (Point3D[2] - z_mean))])
    #print(Common_ArbitraryPoint)
    sum_sqr_vals = sum([((Common_ArbitraryPoint[0] - x_mean) * (Common_ArbitraryPoint[0] - x_mean)) + ((Common_ArbitraryPoint[1] - y_mean) * (Common_ArbitraryPoint[1] - y_mean)) + ((Common_ArbitraryPoint[2] - z_mean) * (Common_ArbitraryPoint[2] - z_mean))])
    m = sum_sqr_vals / k

    # compute the uniform scale
    s = math.sqrt(m)

    # compute a scale factor
    sf = 1/s

    #create the locators
    i = 0
    for p in points:
        x = (p[0] - x_mean) / s
        y = (p[1] - y_mean) / s
        z = (p[2] - z_mean) / s
        #loc = cmds.spaceLocator(p=(x, y, z), name=name_str+str(i))
        landmark_positions.append([x, y, z])
        i += 1

    return landmark_positions, x_mean, y_mean, z_mean, sf

def Show_DLIB_LandmarkIndexes(InputDictionary_of_points,size):
    #debug - check all landmark points for unstructured mesh - do they match up with DLIB index scheme
    for Key in InputDictionary_of_points:
        #Input location is in wrapped in vector form [vector((x,y,z))] so use [0][:] to extract the values
        #print a crosshair at dlib landmark location
        if InputDictionary_of_points[Key] is None:
            continue#skip over iteration of loop
        
        if InputDictionary_of_points[Key].OutputPoint is None:
            continue#skip over iteration of loop
        
        bpy.ops.object.empty_add(radius=size, location=(InputDictionary_of_points[Key].OutputPoint[0][:]))
        #print DLib INDEX at landmark location - so we can check the points are legitimate
        bpy.ops.object.text_add(radius=size/2,enter_editmode=False, align='WORLD', location=(InputDictionary_of_points[Key].OutputPoint[0][:]),rotation=(90.0, 0.0, 180.0))
        #select last object (text object) and label the landmark with the dlib index
        ob=bpy.context.object
        ob.data.body = str(Key)
        
        
        

def BruteForce(Vector3D_list,ListLength):
    #find distance of closest two vectors in a list
    #return the two 
    min_val=float('inf')
    Vector1=None
    Vector2=None
    for i in range (ListLength):
        for j in range (i+1,ListLength):
            if (dist(Vector3D_list[i], Vector3D_list[j]) < min_val):
                min_val=dist(Vector3D_list[i],Vector3D_list[j])
                Vector1=Vector3D_list[i]
                Vector2=Vector3D_list[j]
    #poorly handled bug: if one vector supplied then this algorithm doesnt work
    #so just output input vector
    if Vector1 is None or Vector2 is None:
        Vector1=Vector3D_list[0]
        Vector2=Vector3D_list[0]
    return min_val,Vector1,Vector2

def ProjectPoint_Angles(CameraRef_ID, pointxy,Imagesizexy,WorkingData_Ref):
    ###returns angles to project a point from an image plane (camera FOV)
    ##onto a model
    
    #provide the camera details - assumed to be embedded in the reference object
    #provide the image location to get the image dimensions
    #provide the coordinate of point to be projected
    
    #Get physical camera details embedded in object
    #get reference to model
    CameraObject_Reference = bpy.data.cameras[CameraRef_ID]
    
    
    #find view angle
    #theta = 2 * arctan ((imgdim/2)/focal length))
    #((PointX/ImageSizeX)*ViewAngleDeg_Width)- (ViewAngleDeg_Width/2)
    
    #to handle input mesh from photoscan
    ViewAngleDeg_Height=math.degrees(math.atan((CameraObject_Reference.sensor_height/2/CameraObject_Reference.lens))*2)
    ViewAngleDeg_Width=math.degrees(math.atan((CameraObject_Reference.sensor_width/2/CameraObject_Reference.lens))*2)
    
    
    #if input mesh is from OpenMVG - we have a different method to extract the view angle
    #will throw exception if the object doesnt exist and keep previous details
    
    if WorkingData_Ref.FrustrumObject_list is not None:
        #roll through the list of frustrum objects and select object with the same camera id
        for FrustrumObject in WorkingData_Ref.FrustrumObject_list:
            #get associated frustrum object detail
            if FrustrumObject.Name==CameraRef_ID:
                #set view angle from frustrum data
                ViewAngleDeg_Width=FrustrumObject.Triangle_Horiz_FOV_Deg
                ViewAngleDeg_Height=FrustrumObject.Triangle_Vert_FOV_Deg
            
 
    
    #print("--ProjectPoint_Angles--")
    #print(CameraRef_ID)
    #print("ViewAngleDeg_Height deg", ViewAngleDeg_Height)
    #print("ViewAngleDeg_Width deg", ViewAngleDeg_Width)
    
    #calculate angle from centre point of camera to intersect point on image plane
    Rotation_Y=((int(pointxy[0])/int(Imagesizexy[0]))*ViewAngleDeg_Width)- (ViewAngleDeg_Width/2)
    Rotation_X=((int(pointxy[1])/int(Imagesizexy[1]))*ViewAngleDeg_Height)- (ViewAngleDeg_Height/2)
    
    #ProjectPoint_Angles confirmed Test data
    #ViewAngleDeg_Width 15.7540778606036
    #ViewAngleDeg_Height 23.448288708284988
    #lens 55.0
    #sensor_height 22.828250885009766
    #sensor_width 15.218832969665527
    #pointxy (2095, 1855)
    #Imagesizexy (3456,5184)
    #Angle_X 1.6729590783684953
    #Angle_Y -3.333601230325238
    
    #print("ProjectPoint_Angles Test data")
    #print("ViewAngleDeg_Width", ViewAngleDeg_Width)
    #print("ViewAngleDeg_Height", ViewAngleDeg_Height)
    #print("lens", CameraObject_Reference.lens)
    #print("sensor_height", CameraObject_Reference.sensor_height)
    #print("sensor_width", CameraObject_Reference.sensor_width)
    #print("pointxy", pointxy)
    #print("Angle_X", Angle_X)
    #print("Angle_Y", Angle_Y)
    #print(Imagesizexy)
    #TODO fudge factor to correct for specific camera rotation - needs looked at to generalise
    return Rotation_X*-1,Rotation_Y*-1


    

def Project2DPoints_To_3d(WorkingData,CameraPod_ID,GeneralFunctions, ListOfFacialLandmarks_to_ignore=[]):
    ###requires a pre-existing scene with camera objects and a model
    ###input is a class reference holding working data
    ##can also input a list of facial landmark indexes to ignore for accuracy 
    #facial landmarks are 1 - indexed 
    
    ###no error checking is conducted here! 
    
    #project XY points from text file of image, intersecting point through implied image plane (image plane positioned to camera FOV)
    #and intersecting with model

    #set camera ID
    #CameraPod_ID=WorkingData.MainCamera
    #get reference to blender camera object
    CameraObject_Reference = bpy.data.objects[CameraPod_ID]
    #get reference to model
    Unstructured_Reference = bpy.data.objects[WorkingData.UnstructModel]
    #Make instance of debug objects to assist with ray-tracing
    #create debug line
    DebugLine_ID=GeneralFunctions.DebugLine_from_Camera(CameraPod_ID,WorkingData,True)
    #get reference to debug line
    DebugLine_Reference=bpy.data.objects[DebugLine_ID]
    #create a debug origin object to help with ray casting  
    CameraCube_Origin=GeneralFunctions.CreateCube_At_CameraOrigin(CameraPod_ID,WorkingData,AccumlateID=True)
    CameraCube_Reference_Origin=bpy.data.objects[CameraCube_Origin]   
     #create a debug aim object to help with ray casting  
    CameraCube_Aim=GeneralFunctions.CreateCube_At_CameraOrigin(CameraPod_ID,WorkingData,AccumlateID=True)
    CameraCube_Reference_Aim=bpy.data.objects[CameraCube_Aim] 
    #get dimensions of image
    #ImageSize=GeneralFunctions.GetDimensionsOfImageFile(WorkingData.UnstructuredDirectory + CameraPod_ID)
    ImageSize=Get_ImageDims_from_JSON(WorkingData,CameraPod_ID)

    #handle file of coordinates associated with image 
    
    #if JSON file is found for previous processes, switch from default input folder for manual loading of input files to folder
    #specified from JSON report
    if WorkingData.JSON_FaceAnalysisDetails is not None:
         print("Face Analysis report JSON supersedence, switching default input folder from " + WorkingData.UnstructuredDirectory + " to " +  WorkingData.JSON_FaceAnalysisDetails["OutputFolder"])
         WorkingData.UnstructuredDirectory=WorkingData.JSON_FaceAnalysisDetails["OutputFolder"] + "/"
         
    #concatenate folder location and file
    DlibPointsFile=WorkingData.UnstructuredDirectory + CameraPod_ID
    #file file extension
    DlibPointsFile=DlibPointsFile.replace(".jpg",".txt",1)
    DlibPointsFile=DlibPointsFile.replace(".png",".txt",1)
    DlibPointsFile=DlibPointsFile.replace(".bmp",".txt",1)
    #load text file and format coordinates 
    Dlib_Points=[]
    Dlib_Points_formatted=[]
    print("loading facial landmarks file ", DlibPointsFile)
    with open(DlibPointsFile) as file: 
        text = file.read() 
    Dlib_Points=text.split("\n")
    for element in Dlib_Points:
        Dlib_Points_formatted.append(element.split())
        
        
    Temp_2D_Points=[]
    Temp_3D_Points=[]
    Temp_IndexCheck=[]
       
    #roll through coordinates and create debug ray-tracing objects for each
    for indexer, element in enumerate(Dlib_Points_formatted):
    #for element in Dlib_Points_formatted:
        
        #filter out 1-indexed facial landmarks from input list
        if (indexer+1) in ListOfFacialLandmarks_to_ignore:
            #print(CameraPod_ID , " skipping index ", indexer+1)
            continue #just to next iteration of loop
        
        if element!=[]:
            #calculate rotations to draw line between XY point on image plane and camera origin
            Rotation_X,Rotation_Y=ProjectPoint_Angles(CameraPod_ID, element,ImageSize,WorkingData)
            
            #reset debug ray-trace aim object
            GeneralFunctions.ClearTransforms(CameraCube_Aim)
            CameraCube_Reference_Aim.location = CameraObject_Reference.location
            CameraCube_Reference_Aim.rotation_euler = CameraObject_Reference.rotation_euler
            GeneralFunctions.FreezeObject(CameraCube_Aim)
            
            #move rotations of debug line (convert to radians)
            DebugLine_Reference.rotation_euler[0]=Rotation_X*(math.pi/180)
            DebugLine_Reference.rotation_euler[1]=Rotation_Y*(math.pi/180)
            
            #rotate the ray cast debug origin to match projection lines 
            CameraCube_Reference_Origin.rotation_euler=DebugLine_Reference.rotation_euler
            GeneralFunctions.FreezeObject(CameraCube_Origin)
            
            #rotate the ray cast debug aim to match projection lines 
            CameraCube_Reference_Aim.rotation_euler=DebugLine_Reference.rotation_euler
            GeneralFunctions.FreezeObject(CameraCube_Aim)
            
            #move aim object down axis of projected line to debug the ray cast aim
            # one blender unit in x-direction
            vec = mathutils.Vector((0.0, 0.0, -1.0))
            inv = CameraCube_Reference_Aim.matrix_world.copy()
            inv.invert()
            vec_rot = vec @ inv
            CameraCube_Reference_Aim.location = CameraCube_Reference_Aim.location + vec_rot
            
            GeneralFunctions.DeselectAll()
            (RayCast_result, RayCast_location_local,RayCast_location_global, RayCast_normal, RayCast_index)=GeneralFunctions.RayCast_using_objects(WorkingData.UnstructModel,CameraCube_Origin,CameraCube_Aim,2000)
            #print(RayCast_result, RayCast_location_local, RayCast_normal, RayCast_index)
            
            #if projection worked - add to 
            if RayCast_result:
                Temp_2D_Points.append(element)
                Temp_3D_Points.append(RayCast_location_global)
                Temp_IndexCheck.append(indexer+1)
            else:
                Temp_2D_Points.append(None)
                Temp_3D_Points.append(None)
                Temp_IndexCheck.append(indexer+1) 

    #now populate image dictionary with 2d coordinates and associated 3D coordinates
    #WorkingData.ImageCoordsDictionary[CameraPod_ID]=WorkingData.ImageDataTuple(Temp_2D_Points,Temp_3D_Points)
    #tidy up objects
    GeneralFunctions.DeselectAll()
    GeneralFunctions.DeleteObject(CameraCube_Aim)
    GeneralFunctions.DeselectAll()
    GeneralFunctions.DeleteObject(CameraCube_Origin)
    GeneralFunctions.DeselectAll()
    GeneralFunctions.DeleteObject(DebugLine_ID)
    return Temp_2D_Points,Temp_3D_Points,Temp_IndexCheck


def CheckValidityOf2D_3D_Landmarks(WorkingData):
    print("CheckValidityOf2D_3D_Landmarks skipped")
    return
    ##check validity of 2D and 3D points, generally for ensuring failed raytraces don't offset dlib indexes or any other symptom of unexpected errors
    #loop through all camera objects
    #this will give warnings only for debugging
    
    #generate report
    LandMarkIndexCheckDictionary={}
            
    for CamIndexer, CameraObjectID in enumerate(WorkingData.ImageCoordsDictionary):
        #set temporary counters
        temp_2d_count=0
        temp_3d_count=0
        #consistency check should balance out at zero
        ConsistencyCheck=0
        
        
        try: #check 2d coordinates
            for Index2d, object2d in enumerate (WorkingData.ImageCoordsDictionary[CameraObjectID].Dlib_Coords_2D):
                if object2d is None:
                    print("null/failed/broken 2D point in" , CameraObjectID, " at index ", Index2d)
                    ConsistencyCheck=ConsistencyCheck+Index2d
                else:
                    temp_2d_count=temp_2d_count+1
                    
            #check 3d coordinates
            for Index3d,object3d in enumerate(WorkingData.ImageCoordsDictionary[CameraObjectID].Dlib_Coords_3D):
                if object3d is None:
                    print("null/failed/broken 3D point in" , CameraObjectID, " at index", Index3d)
                    ConsistencyCheck=ConsistencyCheck-Index3d
                else:
                    temp_3d_count=temp_3d_count+1
                    
            #generate report
            LandMarkIndexCheckDictionary={}
            
            #check 3d coordinates
            for Index3d,object3d in enumerate(WorkingData.ImageCoordsDictionary[CameraObjectID]):
                if object3d is None:
                    pass
                else:
                    LandMarkIndexCheckDictionary[str(object3d.DLib_Index)]="poop"
                    print(LandMarkIndexCheckDictionary[str(object3d.DLib_Index)])
                    
                    
                    
                    
                    
        except Exception as e:
            print ("Error checking associated 2d/3d points in ",CameraObjectID," ", e)
            pass
        
        if ConsistencyCheck!=0:
            print ("ERROR consistency check associated 2d/3d points in ",CameraObjectID, " not balanced - bad data")
    
    
    



def Collate_3d_Landmarks(WorkingData):
    ReturnDictionary={}
    #Create landmark dictionary of form [key=landmark Index][value = [3D point 1][3d point 2][3d point n]]
    ##build look-up dictionary holding all 3d points and dlib indexes as keys
    for CamIndexer, CameraObjectID in enumerate(WorkingData.ImageCoordsDictionary):
        
        try:
             #for each camera - get list of all dlib points
            List_of_2d_points_for_camera=WorkingData.ImageCoordsDictionary[CameraObjectID].Dlib_Coords_2D
        except:
            print("Collate_3d_Landmarks: facial landmarks have not been associated with camera, ", CameraObjectID)
            continue#skip current iteration of loop
        
        #for each dlib point for camera/image:
        for Index2d, object2d in enumerate (List_of_2d_points_for_camera):
            #2d and 3d points should be twinned
            Current_2d_Point=WorkingData.ImageCoordsDictionary[CameraObjectID].Dlib_Coords_3D[Index2d]
            Associated_3d_point=WorkingData.ImageCoordsDictionary[CameraObjectID].Dlib_Coords_3D[Index2d]
            Dlib_Index_of_Points=WorkingData.ImageCoordsDictionary[CameraObjectID].DLib_Index[Index2d]
            
            #we cull some dlib points
            if Dlib_Index_of_Points> WorkingData.DlibPoint_Cut_off:
                continue#jumps to next iteration
            
            #error check
            if Current_2d_Point is None:
                pass
            else: 
                if Associated_3d_point is None:
                    pass
                else:
                    #we have 2d and 3d point - copy them into dictionary to help with sorting/averaging
                    #check if dictionary key exists for dlib index
                    if not (Dlib_Index_of_Points) in ReturnDictionary:
                        #index does not exist, create class to hold point/average etc
                        Average3DPoints= WorkingData.Average3DPoints_Class()
                        Average3DPoints.Points=[Associated_3d_point]
                        #create key with new class, populate with 3d point
                        ReturnDictionary[(Dlib_Index_of_Points)]=Average3DPoints
                    else:
                        #update dictionary dlib index with 3d point 
                        ReturnDictionary[(Dlib_Index_of_Points)].Points.append(Associated_3d_point)
                        
    return ReturnDictionary       
   
   
   

def Average_3D_Landmark_Position(WorkingData):
    returndictionary = copy.deepcopy(WorkingData.Unstructured_Model_AverageLandmarksDictionary)
    #now have a dictionary holding all 3d points
    #[Key/Dlib Index] [class of points/average/...]
    #For each set of points per dlib index, find closest two 3d points. These are most likely the two best candidates 
    #to create an average point, as other points can be very far off so should be filtered out
    #each value here will be a class containing raw points for the Dlib index, and a few other containers
    for dlibindex in WorkingData.Unstructured_Model_AverageLandmarksDictionary:
        #brute force find pair of vectors with smallest distance - TODO for more efficiency try "divide & conquer" algorithm
        vDistance,Vvec1,Vvec2=BruteForce( returndictionary[dlibindex].Points, len(returndictionary[dlibindex].Points))
        returndictionary[dlibindex].ClosestPoints_and_distance.append([vDistance,Vvec1,Vvec2])
        
    #For each set of points per dlib index, calculate average 3d position of the list of closest points and distances
    for dlibindex in WorkingData.Unstructured_Model_AverageLandmarksDictionary:
        #first pair of closest vectors at [0] - currently only 1 pair of closest vectors
        Vector1=returndictionary[dlibindex].ClosestPoints_and_distance[0][1]
        Vector2=returndictionary[dlibindex].ClosestPoints_and_distance[0][2]
        #create midpoint of the two vectors
        AverageVector=(Vector1+Vector2)/2
        #store in class
        returndictionary[dlibindex].AverageVector.append(AverageVector)
        #store in class
        returndictionary[dlibindex].OutputPoint.append(AverageVector)
        #debug points
        #bpy.ops.object.empty_add(radius=0.1, location=AverageVector)
    return returndictionary





def LoadGenericModel_3DLandmarks(WorkingData):
    #handle file of coordinates associated with image 
    #concatenate folder location and file
    DlibPointsFile=WorkingData.Resources_Folder + WorkingData.GenericHead_Dlib3DCoords
    Dlib_Points=[]
    Dlib_Points_formatted=[]
    Dlib_points_Corrected=[]
    Dlib_points_Dictionary={}
    with open(DlibPointsFile) as file: 
        text = file.read() 
    Dlib_Points=text.split("\n")
    for element in Dlib_Points:
        Dlib_Points_formatted.append(element.split())


    for indexer, point3d in enumerate(Dlib_Points_formatted):
        #create numpy vector
        # creating a 1-D list (Horizontal) 
        if point3d == []:
            print("warning: generic model 3d dlib point empty for index ",indexer)
            Dlib_points_Dictionary[indexer+1]=None#correct dlib landmarking offset
            continue
        
    #    #if you have time try and implement your own rotation matrix
    #    list1 = [float(point3d[0]),float(point3d[1]),float(point3d[2])] 
    #    # creating a 1-D list (Vertical) 
    #    list2 = [[list1[0]], 
    #            [list1[1]], 
    #            [list1[2]]] 
    #   
    #    vector1 = numpy.array(list1)
    #    RotationMatrix = numpy.array(((1, 3,3), (4, 5,3),(4, 5,3)))
    #    NewPos=vector1@RotationMatrix

        #create vector object
        vec = mathutils.Vector((float(point3d[0]),float(point3d[1]),float(point3d[2])))
        #create rotation matrix
        mat_rot = mathutils.Matrix.Rotation(math.radians(-90.0), 4, 'X')
        #matrix multiplication to rotate point around world origin (to correct for offset)
        vec_rotated=vec@mat_rot
        #add corrected position to list - this can be deleted if dictionary format is used
        Dlib_points_Corrected.append(vec_rotated)
        #format as dictionary - use same class we use for unstructured model to keep things consistent
        SavePoint= WorkingData.Average3DPoints_Class()
        SavePoint.OutputPoint=[vec_rotated]
        Dlib_points_Dictionary[indexer+1]=SavePoint #correct dlib landmarking offset
        
    return Dlib_points_Dictionary