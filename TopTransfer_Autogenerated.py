#NOTE - can run in CONSOLE using myModule = bpy.data.texts[0].as_module()
import bpy
import sys
import os
import math
import mathutils
from mathutils import Vector, Matrix
import numpy 
import csv
import string
from collections import namedtuple
import copy
import bmesh
import numpy
import json
import random

#"Photoscan import works fine - just remember to change the fix camera pose section"
        #self.notes2="openmvg - works but have to change scale during uv project as is flipped x/y, also
        #can get bad camera data in which mangles the uv project - so have to swap from pod primary to secondary
        #to try and fix"
        
        #getting focal length of OpenMVG cameras is a bit dodgy
class WorkingData_Class:
    def __init__(self):
        ###object to hold working details used in process
        #self.DirectoryPrefix="F:/CURRENT/BlenderConversion/QuickBacks/Work/"
        self.DirectoryPrefix="H:/CURRENT/PythonStuff/CUSTOM/Deployed_Workflow_Test/SFM_to_Retop/"
        #AUTO GENERATION SECTION - if part of an automated pipeline this sript will be copied and modified
        self.DirectoryPrefix="H:\CURRENT\PythonStuff\CUSTOM\Deployed_Workflow_Test\SFM_to_Retop"
        #END OF AUTO GENERATION SECTION
        self.UnstructuredDirectory=self.DirectoryPrefix
      
        #self.CustomPython_functionsDirectory=self.DirectoryPrefix +"BlenderConversion/Maya_Topology_transfer/"
        self.OutputDirectory=self.DirectoryPrefix +"/RetopologyOutput/"
        self.Resources_Folder=self.DirectoryPrefix +"/Resources/Retopology/"
        self.OutputRetopModelName="Retopped.obj"
        self.OutputJSON_FacialLandmarksVvertices="FaceLandmarks_V_vertices.json"
        self.OpenSourceSFM_CamDataFile="sfm_data_Extrinsics_Views.json"
        
        self.GenericHead=["Generic_Head.obj","Generic_Head.mtl"]#in resources folder
        self.GenericHead_name='Generic_Head'
        
        #self.GenericHead=["Generic_Head.obj","Generic_Head.mtl"]#in resources folder
        #self.GenericHead_name='Generic_Head'
        
        self.GenericHead_Dlib3DCoords="generic_landmarks.xyz"#in resources folder
        self.Head1="model_1"#do we need this?
        self.Unstructured_Model_file="model_dense_mesh_refine_texture.ply"#"model.fbx"#file name of unstructured 3d head scan output of openmvg/mvs reconstruction
        self.Unstructured_CamFrustrums_file="CameraFrustrums.ply"#file name of camera frustums associated with 3D head scan - spelling error here so be careful 
        self.Unstructured_CamFrustrums="CameraFrustrums"#name of mesh - spelling error here warning
        self.UnstructModel="model_dense_mesh_refine_texture"
        self.Di3D_MainCamera="pod2secondary"
        self.prefix_DebugCylinder="cylinder"
        self.prefix_DebugCube="cube"
        self.PHOTOSCAN_CameraSceneObjectSuffix=".jpg"
        #JSON filenames of previous stages of pipeline if running stages automatically - if these files are not found the default file input folder will be used
        self.JSON_SFM_details="SFM_FileDetails.json"
        self.JSON_FaceAnalysisDetails="FaceAnalysisReport.json"
        
        #facial landmark system dlib has 68 points to describe features of a face - we will cull this to 60
        self.DlibPoint_Cut_off=60
        self.DebugLineCounter=0
        self.ImageCoordsDictionary={}
        #object to hold 2d and associated 3d coordinates
        self.ImageDataTuple = namedtuple('ImageDataTuple', ['Dlib_Coords_2D', 'Dlib_Coords_3D','DLib_Index' ])
        self.MainCamera=""
        self.Unstructured_Model_AverageLandmarksDictionary={}
        self.GenericModel_3D_LandmarksDictionary={}
        
        self.UnstructuredModel_Vertex_of_FacialLandmark={}
        self.GenericModel_Vertex_of_FacialLandmark={}
        self.FrustrumObject_list=None#hold data for camera frustrum mesh objects imported from OpenMVG
        
        self.ProjectionMaterial='Material_UVProjection'
        self.CommonDLIB_Index_ForBasisVector=15
        self.BlenderVersion='2.91.0'
        self.AddOns="Photogrammetry addon"
        self.DetectedBlenderExePath=""
        #may need to filter facial landmarking projection according to camera position for most accurate results
        #this is with 1-indexed system - keep an eye on this 
        self.FacialLandmarks_notCentral_faceIndexes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        self.FacialLandmarks_Central_faceIndexes=[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68]
        #generate left and right subsets of the OUTER face index (chinstraps)
        self.FacialLandmarks_notCentral_faceIndexesLEFT=self.FacialLandmarks_notCentral_faceIndexes[0:int(len(self.FacialLandmarks_notCentral_faceIndexes)/2)]
        self.FacialLandmarks_notCentral_faceIndexesRIGHT=self.FacialLandmarks_notCentral_faceIndexes[int(len(self.FacialLandmarks_notCentral_faceIndexes)/2):]

        
    class Average3DPoints_Class:
        def __init__(self):
            self.Points=[]
            self.ClosestPoints_and_distance=[]
            self.AverageVector=[]
            self.OutputPoint=[]
            
    class FrustrumDetails_Class:
        def __init__(self):
            self.Name=None
            self.OriginPosition=None
            self.AverageFrontFacePoint=None
            self.LowerLeft=None
            self.LowerRight=None
            self.UpperLeft=None
            self.UpperRight=None
            self.LowerAngle=None
            self.UpperAngle=None
            self.Triangle_Horiz_Left_length=None
            self.Triangle_Horiz_Right_Length=None
            self.Triangle_Horiz_Front_Length=None
            self.Triangle_Horiz_AngleDiff=None
            self.Triangle_Horiz_FOV_Deg=None
            
            self.Triangle_Vert_Left_length=None
            self.Triangle_Vert_Right_Length=None
            self.Triangle_Vert_Front_Length=None
            self.Triangle_Vert_AngleDiff=None
            self.Triangle_Vert_FOV_Deg=None

#clear input scene - warning might make it difficult to debug script
#bpy.ops.wm.read_factory_settings(use_empty=True)
 
#new instance of class
WorkingData=WorkingData_Class()

#check blender version
if bpy.app.version_string!=WorkingData.BlenderVersion:
    pass
    #raise Exception("Wrong version of Blender for script, script designed for version ", WorkingData.BlenderVersion)
    
#get file path where blender.exe is located 
WorkingData.DetectedBlenderExePath=str(os.path.dirname(os.path.realpath(sys.argv[0]))) 
print(WorkingData.DetectedBlenderExePath)

#import custom math functions
sys.path.append(os.path.abspath(WorkingData.Resources_Folder))
print(WorkingData.Resources_Folder)
import MathFunctions
import GeneralFunctions
import TopTransfer_Functions
# this next part forces a reload in case you edit the source after you first start the blender session
import imp
imp.reload(MathFunctions)
imp.reload(GeneralFunctions)
imp.reload(TopTransfer_Functions)


#get local path of script
print(str(os.path.dirname(os.path.realpath(sys.argv[0]))))
print(os.getcwd())

        


#reset output folder
GeneralFunctions.DeleteFiles_RecreateFolder(WorkingData.OutputDirectory)

WorkingData.JSON_SFM_details,WorkingData.JSON_FaceAnalysisDetails=GeneralFunctions.CheckFor_JSON_PreProcesses(WorkingData,WorkingData.UnstructuredDirectory)

#if JSON file is found for previous processes, switch from default input folder for manual loading of input files to folder
#specified from JSON report
if WorkingData.JSON_SFM_details is not None:
     print("SFM report JSON supersedence, switching default input folder from " + WorkingData.UnstructuredDirectory + " to " +  WorkingData.JSON_SFM_details["OutputFolder"])
     WorkingData.UnstructuredDirectory=WorkingData.JSON_SFM_details["OutputFolder"] + "/"
     

#deselect anything active in Blender
GeneralFunctions.DeselectAll()        


#load unstructured model & camera poses (depending on source either from external file or
#embedded in model.
#for parenting operations it seems to be important to load the cameras first if they are in an external file

#[order important A1/2]
#handle resource input folder if camera objects are in seperate file to imported unstructured model
TopTransfer_Functions.Load_ExternalCameras_if_exist(WorkingData)

#[order important A2/2]
#import unstructured model & camera poses 
bpy.ops.import_mesh.ply(filepath=WorkingData.UnstructuredDirectory + WorkingData.Unstructured_Model_file)

#Get all associated images from PHOTOSCAN - Opensource will need another solution reading the files
for CameraObject in bpy.data.objects:
    #filter out other objects except assumed naming convention of camera objects
    #check is a camera (bad code)
    try:
        #add item to dictionary - with placeholder data for now
        CameraObject.data.shift_x=CameraObject.data.shift_x#hacky test, will throw exception and skip adding object if not a camera
        WorkingData.ImageCoordsDictionary[CameraObject.name]= []
    except:
        pass
        
        

##print debug line from camera object midline
for CameraObjectID in  WorkingData.ImageCoordsDictionary:
    CameraObject_Reference = bpy.data.objects[CameraObjectID]
    #GeneralFunctions.DebugLine_from_Camera(CameraObjectID,WorkingData,False)
    #fix rotation to match the format subsequent operations expect
    #CameraObject_Reference.rotation_euler.rotate_axis('Z',math.pi)
    #CameraObject_Reference.rotation_euler.rotate_axis('X',math.pi)
    #GeneralFunctions.DebugLine_from_Camera(CameraObjectID,WorkingData,False)

            
##default assign first/main camera (method to get first item in a dictionary)
WorkingData.MainCamera = next(iter(WorkingData.ImageCoordsDictionary))
print("Default camera set to ", WorkingData.MainCamera)
#if Di3D system detected set origin to specific camera 
for CameraObjectID in  WorkingData.ImageCoordsDictionary:
    if WorkingData.Di3D_MainCamera.upper() in CameraObjectID.upper() :
            WorkingData.MainCamera = CameraObjectID#WorkingData.Di3D_MainCamera
            print("Di3D detected - origin camera set to", WorkingData.MainCamera)


#Frustrums 1
#Import Camera Frustrums if file exists
#if using OpenMVG/MVS sfm - we need to import camera frustrum models to position camera
#cameras from OpenMVG have a SHIFTx/y parameter non-zero which causes difficulties with subsequent operations (such as project facial landmarking points)
#therefore we try and mitigate this by converting the SHIFT parameter to a rotation 
WorkingData.FrustrumObject_list=GeneralFunctions.Frustrum_MidlineVector_CornerVector(WorkingData.Unstructured_CamFrustrums,WorkingData)

#this object is only loaded if we are importing the working mesh from OpenMVG
#therefore the camera objects can come with internal principal point shift which needs to
#be converted to gross rotation to allow subsequent operations to complete correctly
if (WorkingData.FrustrumObject_list) is not None:
    for FrustrumObject in WorkingData.FrustrumObject_list:
        #get reference to associated camera object
        CameraObject_Reference = bpy.data.objects[FrustrumObject.Name]
        CameraObject_Reference.data.shift_x=0
        CameraObject_Reference.data.shift_y=0
        CameraObject_Reference.rotation_euler[0]=CameraObject_Reference.rotation_euler[0]+(FrustrumObject.Triangle_Vert_AngleDiff/57.2957795)#to radians
        CameraObject_Reference.rotation_euler[1]=CameraObject_Reference.rotation_euler[1]+(FrustrumObject.Triangle_Horiz_AngleDiff/57.2957795)#to radians
        
        #set focal length of camera in FOV unit (default is mm)
        CameraObject_Reference.data.angle=FrustrumObject.Triangle_Vert_FOV_Deg/57.2957795#to radians

    def look_at(obj_camera, point):
        #function to rotate an object to look towards a location
        #authored https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
        loc_camera = obj_camera.matrix_world.to_translation()

        direction = point - loc_camera
        # point the cameras '-Z' and use its 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')

        # assume we're using euler rotation
        obj_camera.rotation_euler = rot_quat.to_euler()
         

#finished frustrums

#draw debug lines from cameras
for indexer, CameraObjectID in  enumerate(WorkingData.ImageCoordsDictionary):
    
    CameraObject_Reference = bpy.data.objects[CameraObjectID]
    #GeneralFunctions.DebugLine_from_Camera(CameraObjectID,WorkingData,False)



##Parent unstructured model and cameras to Pod 2 Primary camera or default main camera (at origin)
#parent = bpy.data.objects.get(WorkingData.MainCamera )
#child = bpy.data.objects.get(WorkingData.UnstructModel)
#child.parent = parent
#child.matrix_parent_inverse = parent.matrix_world.inverted()

#for indexer, CameraObjectID in  enumerate(WorkingData.ImageCoordsDictionary):
#    if CameraObjectID.upper()!=WorkingData.MainCamera.upper():
#        child = bpy.data.objects.get(CameraObjectID)
#        child.parent = parent
#        child.matrix_parent_inverse = parent.matrix_world.inverted()

#        

##deselect anything active in Blender
#GeneralFunctions.DeselectAll()    

##set main camera to origin
#Unstruct_OriginCamera_ref = bpy.data.objects[WorkingData.MainCamera]
#Unstruct_OriginCamera_ref.location=[0,0,0]
#Unstruct_OriginCamera_ref.rotation_euler=[0,0,0]

##TODO correct to blender axis configuration: keep an eye on this
#Unstruct_OriginCamera_ref.rotation_euler[0]=math.pi/180*90


##adjust scale - TODO NOT IN ORIGINAL METHOD
#Unstruct_OriginCamera_ref.scale=[1,1,1]


##remove parent/child lock
##deselect anything active in Blender
#GeneralFunctions.DeselectAll()
##ensure object is selected
#GeneralFunctions.SelectObject(WorkingData.MainCamera)
##remove parenting but keep associated transforms
#bpy.ops.object.select_grouped(type='CHILDREN_RECURSIVE')
#bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')




#get object centre of unstructured model and set pivot
#deselect anything active in Blender
GeneralFunctions.DeselectAll()
#ensure object is selected
GeneralFunctions.SelectObject(WorkingData.UnstructModel)
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')


#parent all cameras to unstructured model
parent = bpy.data.objects.get(WorkingData.UnstructModel)
for CameraObjectID in  WorkingData.ImageCoordsDictionary:
    child = bpy.data.objects.get(CameraObjectID)
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()


#set unstructured model to origin
Unstruct_ref = bpy.data.objects[WorkingData.UnstructModel]
Unstruct_ref.location=[0,0,0]


##correct pitch of unstructured model
##get reference to model
#Unstructured_Reference = bpy.data.objects[WorkingData.UnstructModel]
##get number of vertices
#UnstructuredModel_no_of_Vertices=len(Unstructured_Reference.data.vertices)
##get local and global vertices 
#Unstructured_vertices_local= [Unstructured_Reference.data.vertices[i].co for i in range (UnstructuredModel_no_of_Vertices)]
#Unstructured_vertices_global= [Unstructured_Reference.matrix_world @ Unstructured_Reference.data.vertices[i].co for i in range (UnstructuredModel_no_of_Vertices)]
##get correction angle for pitch of unstructured model
#theta=(MathFunctions.getRotationAngleFromProfile(Unstructured_vertices_global,[1,0],2,1))#different axis indices
##apply correction angle to unstructured model
#Unstructured_Reference.rotation_euler[0]=Unstruct_ref.rotation_euler[0]+(theta)




#De-parent all cameras from unstructured mesh
#deselect anything active in Blender
GeneralFunctions.DeselectAll()
#ensure object is selected
GeneralFunctions.SelectObject(WorkingData.UnstructModel)
#remove parenting but keep associated transforms
bpy.ops.object.select_grouped(type='CHILDREN_RECURSIVE')
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    
#apply deltas from origin so object local space is origin
GeneralFunctions.FreezeObject(WorkingData.UnstructModel)


#if JSON file is found for previous processes, switch from default input folder for manual loading of input files to folder
#specified from JSON report
if WorkingData.JSON_FaceAnalysisDetails is not None:
     print("Face Analysis report JSON supersedence, switching default input folder from " + WorkingData.UnstructuredDirectory + " to " +  WorkingData.JSON_FaceAnalysisDetails["OutputFolder"])
     WorkingData.UnstructuredDirectory=WorkingData.JSON_FaceAnalysisDetails["OutputFolder"] + "/"

#spin through each camera, locate associated image and associated text file of dlib coordinates
#for each 2d coordinate calculate deflection (using camera instrinsics from the model file) and
#ray-trace onto the model surface to create a 3D point
for CameraObjectID in  WorkingData.ImageCoordsDictionary:
    #Project2DPoints_To_3d returns two lists, therefore we need "*" to unpack them into the VALUE
    #of the dictionary key, which is organised as a named tuple to make it easier to understand
    #can also input a list of facial landmarks to ignore - this is used to improv accuracy for instance if just using one camera for 
    #mid-face points which can be subject to averaging error
    
    #for better accuracy - project mid-face landmarks from one or two cameras
    #occasionaly a camera best positioned for mid-face landmarks can give bad results - maybe have to test
    #camera position for validity somehow
    
    CameraObject_Reference = bpy.data.objects[CameraObjectID]
    #GeneralFunctions.DebugLine_from_Camera(CameraObjectID,WorkingData,False)
    
    #odd results can come from a camera not in correct position and projected points missing target
    if "pod2".lower() in CameraObjectID.lower():
        #ignore list - only process mid-face facial landmarks only
        WorkingData.ImageCoordsDictionary[CameraObjectID]=WorkingData.ImageDataTuple(*TopTransfer_Functions.Project2DPoints_To_3d(WorkingData,CameraObjectID,GeneralFunctions,ListOfFacialLandmarks_to_ignore=WorkingData.FacialLandmarks_notCentral_faceIndexes))
    
    
    #get remaining landmarks (not mid-face) from other cameras
    if  "pod3".lower() in CameraObjectID.lower():
        #ignore list - only process chin-strap facial landmarks only - side corresponding with camera pose
        WorkingData.ImageCoordsDictionary[CameraObjectID]=WorkingData.ImageDataTuple(*TopTransfer_Functions.Project2DPoints_To_3d(WorkingData,CameraObjectID,GeneralFunctions,ListOfFacialLandmarks_to_ignore=(WorkingData.FacialLandmarks_Central_faceIndexes+WorkingData.FacialLandmarks_notCentral_faceIndexesLEFT)))
    
    #get remaining landmarks (not mid-face) from other cameras
    if  "pod1".lower() in CameraObjectID.lower():
        #ignore list - only process chin-strap facial landmarks only - side corresponding with camera pose
        WorkingData.ImageCoordsDictionary[CameraObjectID]=WorkingData.ImageDataTuple(*TopTransfer_Functions.Project2DPoints_To_3d(WorkingData,CameraObjectID,GeneralFunctions,ListOfFacialLandmarks_to_ignore=(WorkingData.FacialLandmarks_Central_faceIndexes+WorkingData.FacialLandmarks_notCentral_faceIndexesRIGHT)))




#check validity and association of 2d and 3d landmarks of unstructured model
TopTransfer_Functions.CheckValidityOf2D_3D_Landmarks(WorkingData)

#Create unstructured model landmark dictionary of form [key=landmark Index][value = [3D point 1][3d point 2][3d point n]]
WorkingData.Unstructured_Model_AverageLandmarksDictionary=TopTransfer_Functions.Collate_3d_Landmarks(WorkingData)

#find average position of all 3D landmark points projected upon unstructured model             
WorkingData.Unstructured_Model_AverageLandmarksDictionary=TopTransfer_Functions.Average_3D_Landmark_Position(WorkingData)

#debug show facial landmarks
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=0.0003)

#get the closest vertex per facial landmark. This allows us to track the facial landmarks during deformation or mesh movement
#warning: closest vertex to facial landmark will have associated translation error due to vertex density
WorkingData.UnstructuredModel_Vertex_of_FacialLandmark=TopTransfer_Functions.GetClosestVertex_of_FacialLandmarks(WorkingData.UnstructModel,WorkingData.Unstructured_Model_AverageLandmarksDictionary,GeneralFunctions)

#print "nearest vertex" facial landmarks - NOTE might be error versus projected landmarks - check visually   
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,GeneralFunctions)


def ShrinkWrap_Custom(MeshID,TargetMeshID):
    #deselect anything active in Blender
    GeneralFunctions.DeselectAll()
    #ensure object is selected
    GeneralFunctions.SelectObject(MeshID)
    bpy.ops.object.modifier_add(type='SHRINKWRAP')
    #note inconsistency between input type and automatic ID
    bpy.context.object.modifiers["Shrinkwrap"].target= bpy.data.objects[TargetMeshID]
    bpy.ops.object.modifier_apply(modifier="Shrinkwrap")


def Export_MeshProjected_CameraImage(ID_of_Camera, Path_of_Image, ID_of_Mesh, WorkingData):
    # Clear all nodes in a mat
    def clear_material( material ):
        if material.node_tree:
            material.node_tree.links.clear()
            material.node_tree.nodes.clear()

    
    #create new material
    ProjectionMat = bpy.data.materials.new(name=WorkingData.ProjectionMaterial)
    #clear_material(ProjectionMat)
    ProjectionMat.use_nodes = True
    nodes = ProjectionMat.node_tree.nodes
    links = ProjectionMat.node_tree.links
    nodes.clear()

    # Create Image Texture node
    Node_TextureImage = nodes.new("ShaderNodeTexImage")
    #associate with image
    Node_TextureImage.image = bpy.data.images.load(Path_of_Image)

    # Create the Texture Coordinate node
    Node_tex_coordinate = nodes.new("ShaderNodeTexCoord")
    #create mapping node
    Node_mapping= nodes.new("ShaderNodeMapping")
    #create emission node
    Node_Emission= nodes.new("ShaderNodeEmission")
    
    ##emission/bsdf nodes will have to be swapped in and out to experiment with what works
    #create bsdf principled node - needed to export the baked/textured model
    Node_BsdfPrincipled= nodes.new("ShaderNodeBsdfPrincipled")
    #adjust parameters of bsdf node
    Node_BsdfPrincipled.inputs["Specular"].default_value=0
    Node_BsdfPrincipled.inputs["Roughness"].default_value=0
    Node_BsdfPrincipled.inputs["Sheen Tint"].default_value=0
    
    #create material output node
    Node_MaterialOutput=nodes.new('ShaderNodeOutputMaterial')

    # connect node chain 
    links.new(Node_mapping.inputs["Vector"], Node_tex_coordinate.outputs["UV"])
    links.new(Node_TextureImage.inputs["Vector"], Node_mapping.outputs["Vector"])
    links.new(Node_Emission.inputs["Color"], Node_TextureImage.outputs["Color"])
    links.new(Node_MaterialOutput.inputs["Surface"], Node_Emission.outputs["Emission"])
    


    # set active material for mesh to receive projected image
    Mesh_Reference = bpy.data.objects[ID_of_Mesh]
    Mesh_Reference.active_material = ProjectionMat

    #deselect anything active in Blender
    GeneralFunctions.DeselectAll()
    #ensure object is selected
    GeneralFunctions.SelectObject(ID_of_Mesh)
    
    #add a copy of the default UV Map
    bpy.ops.mesh.uv_texture_add()

    #add the projection modifier
    bpy.ops.object.modifier_add(type='UV_PROJECT')
    #set the UV map to receive projection image
    bpy.context.object.modifiers["UVProject"].uv_layer = "UVMap"
    
    #set the projection object
    bpy.context.object.modifiers["UVProject"].projectors[0].object = bpy.data.objects[ID_of_Camera]
    #set scale to fix image ratio - need to get this figure from the image dimensions
    ImageX,ImageY=TopTransfer_Functions.Get_ImageDims_from_JSON(WorkingData,ID_of_Camera)

    #set scaling - need different scaling depending on imported mesh software
    if (WorkingData.FrustrumObject_list) is not None:#openMVG 
        print("forcing project UV scale - OpenMVG detected")
        bpy.context.object.modifiers["UVProject"].scale_x = ImageX/ImageY#
    else:
        print("forcing project UV scale - Photoscan detected")
        bpy.context.object.modifiers["UVProject"].scale_y = ImageY/ImageX#
    
    #common filename
    filename=WorkingData.OutputDirectory + ID_of_Camera
    
    #change Blender render engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 2
    bpy.context.scene.render.bake.filepath = filename
    bpy.context.scene.render.bake.save_mode = 'EXTERNAL'
    
    #bake texture/UV
    bpy.ops.object.bake(filepath=filename,  save_mode='EXTERNAL')

    #bpy.ops.image.resize(size=(4096, 4096))
    print("resizing projected image - warning not done nicely - maybe better to rescale using pixel ratio")
    bpy.data.images[ID_of_Camera].scale(4096, 4096)
    bpy.data.images[ID_of_Camera].save_render(filename)
    
    #set image node to recently saved image
    Node_TextureImage.image = bpy.data.images.load(filename)
 
 
    #change from EMISSION to BDSF node to leave in a state for obj exporting (emission node not compatible with model viewers)
    #it is not necessary to delete the emission node - this has been tested by loading into Meshroom which does not handle emission nodes
    links.new(Node_BsdfPrincipled.inputs["Base Color"], Node_TextureImage.outputs["Color"])
    links.new(Node_MaterialOutput.inputs["Surface"], Node_BsdfPrincipled.outputs["BSDF"])
    
    
#import structured generic model 
bpy.ops.import_scene.obj(filepath=WorkingData.Resources_Folder + WorkingData.GenericHead[0])

#import structured generic model associated 3D landmarks
WorkingData.GenericModel_3D_LandmarksDictionary = TopTransfer_Functions.LoadGenericModel_3DLandmarks(WorkingData)


#for input mesh, get the closest vertex per facial landmark. This allows us to track the facial landmarks during deformation
#warning: closest vertex to facial landmark will have associated translation error due to vertex density
WorkingData.GenericModel_Vertex_of_FacialLandmark=TopTransfer_Functions.GetClosestVertex_of_FacialLandmarks(WorkingData.GenericHead_name,WorkingData.GenericModel_3D_LandmarksDictionary,GeneralFunctions)
   

def Get_Projected_UV():
    
    for CameraObjectID in  WorkingData.ImageCoordsDictionary:
        if WorkingData.Di3D_MainCamera not in CameraObjectID :
            Export_MeshProjected_CameraImage(CameraObjectID,WorkingData.UnstructuredDirectory + CameraObjectID,WorkingData.GenericHead_name,WorkingData)
    #do front facing/main camera last so is embedded in exported obj
    for CameraObjectID in  WorkingData.ImageCoordsDictionary:
        if WorkingData.Di3D_MainCamera  in CameraObjectID :
            Export_MeshProjected_CameraImage(CameraObjectID,WorkingData.UnstructuredDirectory + CameraObjectID,WorkingData.GenericHead_name,WorkingData)        
    return

#parent all cameras to unstructured model before transforming unstructured and generic models
parent = bpy.data.objects.get(WorkingData.UnstructModel)
for CameraObjectID in  WorkingData.ImageCoordsDictionary:
    child = bpy.data.objects.get(CameraObjectID)
    child.parent = parent
    child.matrix_parent_inverse = parent.matrix_world.inverted()



####experimental
#rotate,scale and translate unstructured mesh onto generic mesh - generic mesh position dictates final position

#get scale difference between unstructured and generic mesh

#apply scale difference to unstructured mesh

#now rotate unstructured mesh to generic mesh - NOTE must be at same scale first 


def GetScaleDifferenceBetween_CorrespondingPoints(List_Facial_PointsA,List_Facial_PointsB,Mesh_A,Mesh_B,GeneralFunctions):
    
#    #get reference to mesh
#    AssociatedMesh = bpy.data.objects.get(MeshID)
#    #move cursor location to mean of facial landmark points
#    bpy.context.scene.cursor.location = [transform_x,transform_y,transform_z]
#    #deselect all objects
#    GeneralFunctions.DeselectAll()
#    #select input mesh object
#    GeneralFunctions.SelectObject(MeshID)
#    #set origin to cursor
#    bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
#    
#    
    
    return
    for LandMark in List_Facial_PointsA:
        if List_Facial_PointsA[LandMark] is None:
            continue#skip this iteration of loop
        if List_Facial_PointsA[LandMark].OutputPoint is None:
            continue#skip this iteration of loop




    #set location to origin
    AssociatedMesh.location.x=0
    AssociatedMesh.location.y=0
    AssociatedMesh.location.z=0
    #apply scale
    AssociatedMesh.scale=AssociatedMesh.scale*Scale_Factor







#stuctured and unstructured objects must be same size to perform alignment
#we want to retain the structured model scale (suffixed with "64mm") as is scaled in Blender units
#to match the average scale of an adult head (interpupilliary distance of 64mm - blender units of 0.064)
def Get_Facial_Landmarks_from_VertexAssociation(InputMeshID,VertexLookup_FaceLandmarks,FacialLandmarks):
    #this function returns global locations of the mesh vertices associated with facial landmarks
    #IE, the input is in form Landmark(x)=Vertex(y) from previous "what vertex is this landmark closest to" association
    #if we have corrupted the positions of the (slightly more accurate) landmarks due to orientation problems,
    #the landmarks associated with the vertices will naturally follow the mesh and any mesh deformation,
    #so we can return the 3d positions and rebuild a corrupted landmark dictionary
    
    #get copy of landmark dictionary
    CopyOfLandMark_Dictionary=copy.deepcopy(FacialLandmarks)
    
    #select generic model
    GeneralFunctions.DeselectAll()
    GeneralFunctions.SelectObject(InputMeshID)

    #get reference to model
    GenericMesh_reference = bpy.data.objects.get(InputMeshID)
    GenericMesh_Num_Vertices=len(GenericMesh_reference.data.vertices.items())
    
    #get local and global vertices 
    Mesh_vertices_local= [GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    Mesh_vertices_global= [GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    
    for Facial_LandMarkKey in FacialLandmarks:
        #Facial_LandMarkKey will be a facial landmark index integer
        #use this as look up to get associated vertex on the mesh
        #this should not be able to fail as the vertex lookup is generated by the facial landmarks
        Associated_Vertex_to_Landmark=VertexLookup_FaceLandmarks[Facial_LandMarkKey]
        #we now have the vertex index associated with the facial landmark
        #facial landmark [Facial_LandMarkKey] is closest to vertex [Associated_Vertex_to_Landmark]
        
        #now get the global position of the vertex
        location=Mesh_vertices_global[Associated_Vertex_to_Landmark]#this is in vector form
        
        #get current item of facial landmark just for illustration
        testVector=FacialLandmarks[Facial_LandMarkKey].OutputPoint[0]
        
        #replace current position of landmark with vertex positioned landmark
        CopyOfLandMark_Dictionary[Facial_LandMarkKey].OutputPoint[0]=location
    
    return CopyOfLandMark_Dictionary

#get scale adjustment needed to scale unstructured to structured, using facial landmarks 
def GetDistance_between_Faciallandmarks(DictionaryOfLandmarks,FacialLandmark1,FacialLandmark2):
    #get distance in blender units between two facial landmarks
    #might effect robustness of process if landmarks do not exist
    #returns a distance in Blender units
    Landmark1=DictionaryOfLandmarks[FacialLandmark1].OutputPoint[0]
    Landmark2=DictionaryOfLandmarks[FacialLandmark2].OutputPoint[0]
    return (Landmark1-Landmark2).length


Unstructured_Sample_Distance=GetDistance_between_Faciallandmarks(WorkingData.Unstructured_Model_AverageLandmarksDictionary,FacialLandmark1=37,FacialLandmark2=43)
Structured_Sample_Distance=GetDistance_between_Faciallandmarks(WorkingData.GenericModel_3D_LandmarksDictionary,FacialLandmark1=37,FacialLandmark2=43)
ScaleAdjustment_needed=Structured_Sample_Distance/Unstructured_Sample_Distance


#scale unstructured mesh to match structured mesh using average of facial landmarks as origin
WorkingData.Unstructured_Model_AverageLandmarksDictionary=TopTransfer_Functions.Scale_Mesh_WithFacialLandmarks_As_Origin(WorkingData.UnstructModel,WorkingData.Unstructured_Model_AverageLandmarksDictionary, WorkingData.CommonDLIB_Index_ForBasisVector,GeneralFunctions,ScaleAdjustment_needed)
#scale structured mesh with scale factor of 1 to set origin as average position of facial landmarks
WorkingData.GenericModel_3D_LandmarksDictionary=TopTransfer_Functions.Scale_Mesh_WithFacialLandmarks_As_Origin(WorkingData.GenericHead_name,WorkingData.GenericModel_3D_LandmarksDictionary,WorkingData.CommonDLIB_Index_ForBasisVector,GeneralFunctions,1)

#refresh facial landmarking
WorkingData.Unstructured_Model_AverageLandmarksDictionary=Get_Facial_Landmarks_from_VertexAssociation(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,WorkingData.Unstructured_Model_AverageLandmarksDictionary)
#WorkingData.GenericModel_3D_LandmarksDictionary=Get_Facial_Landmarks_from_VertexAssociation(WorkingData.GenericHead_name,WorkingData.GenericModel_Vertex_of_FacialLandmark,WorkingData.GenericModel_3D_LandmarksDictionary)


#scale objects using the associated facial landmark points as datasets - for alternatives search for "A Survey of Rigid 3D Pointcloud Registration Algorithms" paper
#this will modify the origin of each mesh to be the mean position of their facial landmark points
#returns updated dictionary of dlib facial landmarks - scales/rotates mesh destructively
#SCALE ONLY
#WorkingData.Unstructured_Model_AverageLandmarksDictionary=TopTransfer_Functions.ProcrustesAlign_ScaleOnly(WorkingData.UnstructModel,WorkingData.Unstructured_Model_AverageLandmarksDictionary, WorkingData.CommonDLIB_Index_ForBasisVector,GeneralFunctions)
#WorkingData.GenericModel_3D_LandmarksDictionary=TopTransfer_Functions.ProcrustesAlign_ScaleOnly(WorkingData.GenericHead_name,WorkingData.GenericModel_3D_LandmarksDictionary,WorkingData.CommonDLIB_Index_ForBasisVector,GeneralFunctions)


#all 4 facial landmarks look good at this point 

#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.GenericModel_3D_LandmarksDictionary,size=1)
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=1)
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,GeneralFunctions)
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.GenericHead_name,WorkingData.GenericModel_Vertex_of_FacialLandmark,GeneralFunctions)








#at this point using debug facial landmark functions both meshes have landmarks in the correct place
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=1)

#align unstructured mesh to the generic mesh using rigid 3D rotation solution
#this will only work if the mesh objects and the associated facial landmark have the same origin (facial landmark centroid)
#and are same scale
#RotationMatrix, Translation = TopTransfer_Functions.RotateAlign_sets_of_3Dpoints(WorkingData.GenericHead_name,WorkingData.Unstructured_Model_AverageLandmarksDictionary,WorkingData.GenericModel_3D_LandmarksDictionary)
RotationMatrix, Translation = TopTransfer_Functions.RotateAlign_sets_of_3Dpoints(WorkingData.UnstructModel,WorkingData.Unstructured_Model_AverageLandmarksDictionary,WorkingData.GenericModel_3D_LandmarksDictionary)

#rotate associated facial landmarks of aligned mesh with output from rigid align solution
#NOTE this function works with previous logic whereby the generic mesh is aligned to the unstructured, but
#it is not understood why this rotation does not work in the reverse manner, yet aligns the mesh itself to the generic mesh. 
TopTransfer_Functions.Destructive_RotateDictionaryOfLandmarks(WorkingData.Unstructured_Model_AverageLandmarksDictionary,RotationMatrix)

##debug - label facial landmarks
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.GenericModel_3D_LandmarksDictionary,size=1)
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=1)

#DEBUG
#print "nearest vertex" facial landmarks - NOTE might be error versus projected landmarks - check visually   
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,GeneralFunctions)
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.GenericHead_name,WorkingData.GenericModel_Vertex_of_FacialLandmark,GeneralFunctions)

#ERROR
#at this stage:
#Unstructured projected facial landmarks: BROKEN - unknown why rotation matrix did not rotate these correctly
#Unstructured vertex-associated facial landmarks: OK
#Generic projected facial landmarks: OK
#Generic vertex-associated facial landmarks: OK
#ERROR

#WORK-AROUND to position unstructured facial landmarks
#convert the second set of facial landmarks (associated with mesh vertexes) and decompose back to global 3d positions
#this might result in loss of accuracy




    
WorkingData.Unstructured_Model_AverageLandmarksDictionary=Get_Facial_Landmarks_from_VertexAssociation(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,WorkingData.Unstructured_Model_AverageLandmarksDictionary)


#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.GenericModel_3D_LandmarksDictionary,size=1)
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=1)
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.UnstructModel,WorkingData.UnstructuredModel_Vertex_of_FacialLandmark,GeneralFunctions)
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.GenericHead_name,WorkingData.GenericModel_Vertex_of_FacialLandmark,GeneralFunctions)



#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=1)







#set MATERIAL view
for area in bpy.context.screen.areas: 
    if area.type == 'VIEW_3D':
        for space in area.spaces: 
            if space.type == 'VIEW_3D':
                space.shading.type = 'MATERIAL'
                space.shading.use_scene_lights=True
                space.shading.use_scene_world=True
                
#deselect anything active in Blender
GeneralFunctions.DeselectAll()               
#add light object      
bpy.ops.object.light_add(type='SUN', radius=100, align='WORLD', location=(112.85299682617188, -176.36099243164062,139.6), scale=(1, 1, 1))
bpy.context.object.data.energy = 20
bpy.context.object.rotation_euler[0] = 0.420624


#add debug camera
bpy.ops.object.camera_add(location = [112.85299682617188, -176.36099243164062, 0.0], rotation = [1.5707963705062866, 0.0, -5.69326353073120],enter_editmode=False)
context = bpy.context
scene = context.scene
scene.camera = context.object
#NOTE - will have automatic name "Camera"
DebugCamName="Camera"
DebugCam=bpy.data.objects.get(DebugCamName)
DebugCam.data.lens=14.0#set focal length to frame subject and deformation sequence


#deselect anything active in Blender
GeneralFunctions.DeselectAll()
#ensure object is selected
GeneralFunctions.SelectObject(DebugCamName)

#move unstructured mesh out the way for pictures
UnstructMesh_reference = bpy.data.objects.get(WorkingData.UnstructModel)
TempLocation=UnstructMesh_reference.location[:]#[:] makes a copy rather than a reference of the object
UnstructMesh_reference.location=[2000,0,0]
#bpy.ops.view3d.object_as_camera()
#pre-deform the generic mesh, to conform to the unstructured mesh using the corresponding facial landmark points as deform anchors and targets
ImageCount=0
for i in range(40, 5,-5):
    bpy.context.scene.render.image_settings.file_format='JPEG'
    filename=WorkingData.OutputDirectory + "DeformStep" + str(ImageCount)
    bpy.context.scene.render.filepath = filename + ".jpg"
    bpy.ops.render.render(use_viewport = True, write_still=True)
    TopTransfer_Functions.PreOptimise_Shrinkwrap_ProportionalDeform(GeneralFunctions,WorkingData.GenericHead_name,WorkingData,DifferenceDivison=i,BrushSize=int(i*3),ReverseScan=False)
    ImageCount=ImageCount+1
#take another image
filename=WorkingData.OutputDirectory + "DeformStep" + str(ImageCount)
bpy.ops.render.render(use_viewport = True, write_still=True)
ImageCount=ImageCount+1
#move unstructured mesh back
UnstructMesh_reference.location=TempLocation
#once closest vertices to facial landmarks have been associated (GetClosestVertex_of_FacialLandmarks)- use this function to visualise and debug deformation
#TopTransfer_Functions.DebugPrint_Landmarks_using_NearestVertex(WorkingData.GenericHead_name,WorkingData.GenericModel_Vertex_of_FacialLandmark,GeneralFunctions)
#TopTransfer_Functions.Show_DLIB_LandmarkIndexes(WorkingData.Unstructured_Model_AverageLandmarksDictionary,size=3)



#tweak position of generic head for debugging 
StructMesh_reference = bpy.data.objects.get(WorkingData.GenericHead_name)
#StructMesh_reference.location=[-1.82,0,9.09]


#deselect anything active in Blender
GeneralFunctions.DeselectAll()
#ensure object is selected
GeneralFunctions.SelectObject(WorkingData.GenericHead_name)
bpy.ops.object.modifier_add(type='SHRINKWRAP')
#note inconsistency between input type and automatic ID
bpy.context.object.modifiers["Shrinkwrap"].target= bpy.data.objects[WorkingData.UnstructModel]

#smooth out vertices (almost same as maya "relax" operation)
bpy.ops.object.modifier_add(type='CORRECTIVE_SMOOTH')
bpy.context.object.modifiers["CorrectiveSmooth"].factor = 1#0.460
bpy.context.object.modifiers["CorrectiveSmooth"].iterations = 2
bpy.context.object.modifiers["CorrectiveSmooth"].scale = 0

#apply modifiers in correct order
bpy.ops.object.modifier_apply(modifier="Shrinkwrap")
bpy.ops.object.modifier_apply(modifier="UVProject")
bpy.ops.object.modifier_apply(modifier="CorrectiveSmooth")



#DEBUG image
#move mesh out the way of camera
TempLocation=UnstructMesh_reference.location[:]#[:] makes a copy rather than a reference of the object
UnstructMesh_reference.location=[2000,0,0]
#take debug image
bpy.context.scene.render.filepath = filename +"End" + ".jpg"
bpy.ops.render.render(use_viewport = True, write_still=True)
ImageCount=ImageCount+1
#move mesh back
UnstructMesh_reference.location=TempLocation




#project texture from cameras onto deformed generic mesh
Get_Projected_UV()


#De-parent all cameras from unstructured mesh
#deselect anything active in Blender
GeneralFunctions.DeselectAll()
#ensure object is selected
GeneralFunctions.SelectObject(WorkingData.UnstructModel)
#remove parenting but keep associated transforms
bpy.ops.object.select_grouped(type='CHILDREN_RECURSIVE')
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

#generic head will be correct scale but wrong magnitude
AssociatedMesh = bpy.data.objects.get(WorkingData.GenericHead_name)
AssociatedMesh.scale=AssociatedMesh.scale*0.001

#export model
#currently doesnt seem to play nicely opened in anything other than BLender - not an issue at present - but problem
#is related to saving the correct image (or a dummy image) into the Image File shader node.. Sometimes this is automatically  
#renamed with a ".00n" suffix which breaks any image association
GeneralFunctions.DeselectAll()
GeneralFunctions.SelectObject(WorkingData.GenericHead_name)
#create random number as output filename to save having to rename items during volume testing
WorkingData.OutputRetopModelName= str(round(random.random()*100000000000)) + ".obj"
bpy.ops.export_scene.obj(filepath=WorkingData.OutputDirectory + WorkingData.OutputRetopModelName,use_selection=True)

#export facial landmarks associated with vertex of models as JSON
#initialise dictionary
Export_FacialLandmarks={}
#create key value pairs for both models (structured and unstructured)
Export_FacialLandmarks["UnstructuredModel_Vertex_of_FacialLandmark"]=copy.deepcopy(WorkingData.UnstructuredModel_Vertex_of_FacialLandmark)
Export_FacialLandmarks["GenericModel_Vertex_of_FacialLandmark"]=copy.deepcopy(WorkingData.GenericModel_Vertex_of_FacialLandmark)
filepath=WorkingData.OutputDirectory + WorkingData.OutputJSON_FacialLandmarksVvertices

with open(filepath, 'w') as jsonFile:
    json.dump(Export_FacialLandmarks, jsonFile)
