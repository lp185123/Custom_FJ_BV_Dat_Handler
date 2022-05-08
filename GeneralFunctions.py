import bpy
import math
import sys
import os
import copy
import numpy
import mathutils
from mathutils import Vector, Matrix
import json
import shutil

def DeselectAll():
###Ensure nothing is selected 
    #two methods to deslect all objects - see which one works
    try:
            
        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.context.selected_objects:
            #print ("Objects still selected: " + obj)
            obj.select_set(False)
        #deselect anything active
        bpy.context.active_object.select_set(False)
    except:
        pass

def JSON_Open(InputFilePath):
    ##open a JSON file
    #TODO this is pretty sketchy
    try:
        if os.path.exists(InputFilePath):
            # Opening JSON file
            fObj = open(InputFilePath,)

            # It returns JSON object as dictionary
            ogdata = json.load(fObj)
            # Closing file
            fObj.close()
            return ogdata

    except Exception as e:
        print("JSON_Open error attempting to open json file " + str(InputFilePath) + " " + str(e))
        raise Exception (e)
    print("JSON_Open: " + InputFilePath + " does not exist, skipping")
    return None

def CheckFor_JSON_PreProcesses(WorkingData,JSON_FileLocationFolder):
    #Previous processes of automated pipeline (sfm stage and face analysis stage) may be present - if so 
    #check for JSON files which will provide locations of input file
    #this is to faciliate portability and modularity 
    print("Checking for SFm and Face Analysis JSON reports if they exist in folder: " + JSON_FileLocationFolder)
    JSON_SFMReport=JSON_Open(JSON_FileLocationFolder +"/" + WorkingData.JSON_SFM_details)
    JSON_FaceAnalysisReport=JSON_Open(JSON_FileLocationFolder +"/" + WorkingData.JSON_FaceAnalysisDetails)
    #XOR files, if both dont exist user most likely is not continuning a staged workflow
    if (JSON_SFMReport is None) ^ (JSON_FaceAnalysisReport is None):
        print("Need both SFM and Face Analysis stage JSON files to import input data, files expected here:")
        print(JSON_FileLocationFolder +"/" + WorkingData.JSON_SFM_details)
        print(JSON_FileLocationFolder +"/" + WorkingData.JSON_FaceAnalysisDetails)
    
    #if we have both files, we can now extract input file locations 
    if (JSON_SFMReport is not None) and (JSON_FaceAnalysisReport is not None):
        print("SFM and Face Analysis JSON files found, extracting file locations for input data")
        print("Unstructured Face Scan and associated files expected at " + JSON_SFMReport["OutputFolder"])
        print("Face landmarking files and associated files expected at " + JSON_FaceAnalysisReport["OutputFolder"])
    
    return JSON_SFMReport,JSON_FaceAnalysisReport


def SelectObject(object_ID):
    objectToSelect = bpy.data.objects[object_ID]
    objectToSelect.select_set(True)
    bpy.context.view_layer.objects.active = objectToSelect

def DeleteObject(object_ID):
    objectToSelect = bpy.data.objects[object_ID]
    objectToSelect.select_set(True)
    bpy.context.view_layer.objects.active = objectToSelect
    bpy.ops.object.delete(use_global=False, confirm=False)

def look_at(obj_camera, point):
    #function to rotate an object to look towards a location
    #authored https://blender.stackexchange.com/questions/5210/pointing-the-camera-in-a-particular-direction-programmatically
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
def FreezeObject(ObjectID):
    #"freeze" unstructured mesh (set all deltas to zero)
    #deselect anything active in Blender
    DeselectAll()
    #ensure object is selected
    SelectObject(ObjectID)
    bpy.ops.object.transforms_to_deltas(mode='ALL')


def CreateCube_At_Location(Location, Scale, WorkingDataReference, AccumlateID=True):
    ###Create a cube at location
    # deselect anything active in Blender
    DeselectAll()
    # Create a simple cube - note - cuboids tend to work better with subsequent operations
    # bpy.ops.mesh.primitive_cylinder_add(radius = 1)
    bpy.ops.mesh.primitive_cube_add(size=Scale, enter_editmode=False, align='WORLD', location=Location)

    # Get the cylinder object and rename it.
    newcube = bpy.context.object
    newcube.name = WorkingData.prefix_DebugCube + ID_of_Camera

    # option to add unique ID by incrementing a global figure
    if AccumlateID == True:
        newcube.name = newcube.name + str(WorkingDataReference.DebugLineCounter)
        WorkingDataReference.DebugLineCounter = WorkingDataReference.DebugLineCounter + 1

    FreezeObject(newcube.name)

    return newcube.name


def CreateCube_At_CameraOrigin(ID_of_Camera, WorkingDataReference, AccumlateID=True):
    ###Create a cube at the camera origin, used for operations such as RAY CAST which needs a mesh to operate from
    # draw target lines from center of cameras to check positioning VS images
    # input string ID of camera
    TempCameraRef = bpy.data.objects[ID_of_Camera]
    # deselect anything active in Blender
    DeselectAll()
    # Create a simple cube - note - cuboids tend to work better with subsequent operations
    # bpy.ops.mesh.primitive_cylinder_add(radius = 1)
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    # Get the cylinder object and rename it.
    newcube = bpy.context.object
    newcube.name = WorkingDataReference.prefix_DebugCube + ID_of_Camera

    # option to add unique ID by incrementing a global figure
    if AccumlateID == True:
        newcube.name = newcube.name + str(WorkingDataReference.DebugLineCounter)
        WorkingDataReference.DebugLineCounter = WorkingDataReference.DebugLineCounter + 1

    # Change the location of the cylinder.
    newcube.location = TempCameraRef.location
    newcube.rotation_euler = TempCameraRef.rotation_euler
    newcube.scale[0] = 0.1
    newcube.scale[1] = 0.1
    newcube.scale[2] = 0.1
    FreezeObject(newcube.name)

    # solidify object - helps with subsequent operation "boolean" intersection
    # deselect anything active in Blender
    DeselectAll()
    # ensure object is selected
    SelectObject(newcube.name)
    # solidify may help with further operations such as "BOOLEAN"
    bpy.ops.object.modifier_apply(modifier="Solidify")

    return newcube.name


def ClearTransforms(ObjectID):
    # unfreeze mesh (remove transforms)
    # deselect anything active in Blender
    DeselectAll()
    # ensure object is selected
    SelectObject(ObjectID)
    bpy.ops.object.location_clear(clear_delta=True)
    bpy.ops.object.rotation_clear(clear_delta=True)
    bpy.ops.object.origin_clear()


def RayCast_using_objects(TargetObject, Origin_of_raycast_as_objectID, Aiming_OfRaycast_as_objectID, distance_of_cast):
    ##return raycast information, casting from originObject towards aimingObject by calculating the direction vector
    ###in target object space

    # ray casting
    # check if in correct mode for ray-casting to work - TODO may not be necessary but keep for now until sure
    if bpy.context.mode != 'OBJECT':
        raise Exception("Please switch to object mode")
    # get reference to objects
    Ray_Origin = bpy.data.objects[Origin_of_raycast_as_objectID]
    Ray_Aim_Point = bpy.data.objects[Aiming_OfRaycast_as_objectID]
    TargetObject_ref = bpy.data.objects[TargetObject]
    # get translation of objects
    global_Ray_Origin = Ray_Origin.matrix_world.translation
    global_Ray_Aim_Point = Ray_Aim_Point.matrix_world.translation
    # convert from global space to local space of object B
    local_Ray_Origin = TargetObject_ref.matrix_world.inverted() @ global_Ray_Origin
    local_Ray_Aim_Point = TargetObject_ref.matrix_world.inverted() @ global_Ray_Aim_Point

    LookDirection = (local_Ray_Aim_Point - local_Ray_Origin).normalized()

    (result, location_local, normal, index) = TargetObject_ref.ray_cast(local_Ray_Origin, LookDirection,
                                                                        distance=distance_of_cast)
    location_global = -1
    if result:
        #bpy.ops.object.empty_add(radius=0.1, location=TargetObject_ref.matrix_world @ location_local)
        location_global = TargetObject_ref.matrix_world @ location_local

    return (result, location_local, location_global, normal, index)


def RayCast_using_direction(TargetObject, Origin_of_raycast_as_objectID, AimDirection, distance_of_cast):
    print("WARNING: NOT TESTED")
    ##return raycast information, casting from originObject towards AimDirection (provided in global coordinates)
    ###in target object space

    # ray casting
    # check if in correct mode for ray-casting to work - TODO may not be necessary but keep for now until sure
    if bpy.context.mode != 'OBJECT':
        raise Exception("Please switch to object mode")
    # get reference to objects
    Ray_Origin = bpy.data.objects[Origin_of_raycast_as_objectID]
    TargetObject_ref = bpy.data.objects[TargetObject]
    # get translation of objects
    global_Ray_Origin = Ray_Origin.matrix_world.translation
    # convert from global space to local space of object B
    local_Ray_Origin = TargetObject_ref.matrix_world.inverted() @ global_Ray_Origin

    LookDirection = -(TargetObject_ref.matrix_world.inverted() @ AimDirection).normalized()

    (result, location_local, normal, index) = TargetObject_ref.ray_cast(local_Ray_Origin, LookDirection,
                                                                        distance=distance_of_cast)
    location_global = -1
    if result:
        bpy.ops.object.empty_add(radius=1000.0, location=TargetObject_ref.matrix_world @ location_local)
        location_global = TargetObject_ref.matrix_world @ location_local

    return (result, location_local, location_global, normal, index)



def DebugLine_from_Camera(ID_of_Camera, WorkingDataReference, AccumlateID=True):
    # draw target lines from center of cameras to check positioning VS images
    # input string ID of camera
    TempCameraRef = bpy.data.objects[ID_of_Camera]
    # deselect anything active in Blender
    DeselectAll()
    # Create a simple cube - note - cuboids tend to work better with subsequent operations
    # bpy.ops.mesh.primitive_cylinder_add(radius = 1)
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0))

    # Get the cylinder object and rename it.
    cyl = bpy.context.object
    cyl.name = WorkingDataReference.prefix_DebugCylinder + ID_of_Camera

    # option to add unique ID by incrementing a global figure
    if AccumlateID == True:
        cyl.name = cyl.name + str(WorkingDataReference.DebugLineCounter)
        WorkingDataReference.DebugLineCounter = WorkingDataReference.DebugLineCounter + 1

    # Change the location of the cylinder.
    cyl.location = TempCameraRef.location
    cyl.rotation_euler = TempCameraRef.rotation_euler
    cyl.scale[0] = 0.001
    cyl.scale[1] = 0.001
    cyl.scale[2] = 1000
    FreezeObject(cyl.name)

    # solidify object - helps with subsequent operation "boolean" intersection
    # deselect anything active in Blender
    DeselectAll()
    # ensure object is selected
    SelectObject(cyl.name)
    # solidify may help with further operations such as "BOOLEAN"
    bpy.ops.object.modifier_apply( modifier="Solidify")

    return cyl.name

def CorrectImportedCameraPose(ID_of_Camera):
    ### Cameras imported from agisoft are flipped, repair pose interpretation
    ### input string ID of camera
    TempCameraRef = bpy.data.objects[ID_of_Camera]
    TempCameraRef.rotation_euler[0]=TempCameraRef.rotation_euler[0]+math.pi
    #deselect anything active in Blender
    DeselectAll()
    #ensure object is selected
    SelectObject(ID_of_Camera)
    #rotate Z - do not understand why this doesnt work with offsets - TODO 
    bpy.ops.transform.rotate(value=math.pi, orient_axis='Z', orient_type='LOCAL')
    #print("CorrectImportedCameraPose in generalfunction undoing camera rotation for custom sfm")
    #bpy.ops.transform.rotate(value=math.pi, orient_axis='Z', orient_type='LOCAL')

def DistanceBtwn_Vectors(Vec1,Vec2):
    Distance=math.sqrt((Vec1[0] - Vec2[0]) ** 2 + (Vec1[1] - Vec2[1]) ** 2 + (Vec1[2] - Vec2[2]) ** 2)
    return Distance




def angle (a, b, c):
    return math.degrees(math.acos((c**2 - b**2 - a**2)/(-2.0 * a * b)))




def Frustrum_MidlineVector_CornerVector(MeshID,WorkingData):
    #CustomSFM pipeline can output camera frustrums, which can be used to 
    #get camera rotation (at time of development the output camera locations from JSON file work but
    #the rotations are slightly off due to modelling of the camera slew which doesnt translate well into Blender)
    
    #returns midline vector of frustrum (middle of square end) and a corner vector (of square end)
    
    frustrumfilepath=WorkingData.UnstructuredDirectory + WorkingData.Unstructured_CamFrustrums_file
    if os.path.exists(frustrumfilepath)==False:
        print("cannot load frustrums - ignore error if not using OpenMVGMVS sfm ", frustrumfilepath)
        raise Exception
        
    #import camera frustrums output from customSFM to help us align cameras if using this pipeline
    bpy.ops.import_mesh.ply(filepath=frustrumfilepath)
    #bpy.ops.import_scene.fbx(filepath=frustrumfilepath)

    #get reference to model
    GenericMesh_reference = bpy.data.objects.get(MeshID)
    GenericMesh_Num_Vertices=len(GenericMesh_reference.data.vertices.items())  
    
    #get local and global vertices 
    Mesh_vertices_local= [GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    Mesh_vertices_global= [GenericMesh_reference.matrix_world @ GenericMesh_reference.data.vertices[i].co for i in range (GenericMesh_Num_Vertices)]
    
    #roll through frustrum vertices
    #each frustrum is assumed to be 5 points 
    #use custom class to hold all details of frustrum
    Counter=0#manual for loop counter
    VerticePerFrustrum=5#we know how many vertices each frustrum should have
    FrustrumList=[]#hold instances of frustrum data class in list
    TempFrustrumData=WorkingData.FrustrumDetails_Class()#temporary frustrum dataclass
    for index, iVertice in enumerate( Mesh_vertices_global):
        Counter=Counter+1
        if Counter % VerticePerFrustrum == 0:
            TempFrustrumData.OriginPosition=Mesh_vertices_global[index-4]#check
            TempFrustrumData.LowerLeft=Mesh_vertices_global[index]#check
            TempFrustrumData.LowerRight=Mesh_vertices_global[index-1]#check
            TempFrustrumData.UpperLeft=Mesh_vertices_global[index-3]#check
            TempFrustrumData.UpperRight=Mesh_vertices_global[index-2]#check
            NewFrustrumData=WorkingData.FrustrumDetails_Class()
            NewFrustrumData=copy.deepcopy(TempFrustrumData)#make deep copy to create new instance instead of reference
            FrustrumList.append(NewFrustrumData)
    
    #label frustrum object with associated camera view
    #by happenstance, due to loading sequence, the list of frustrum objects and list of camera objects are matched by index
    for indexer, CameraObjectID in  enumerate(WorkingData.ImageCoordsDictionary):
        FrustrumList[indexer].Name=str(CameraObjectID)
            
    #at this point should have a list of objects, each object loaded with frustrum information
    #now get average positon of the 4 vertices which make up the rectangular frustrum face
    for FrustumInstance in FrustrumList:
        AverageVector=mathutils.Vector((0.0, 0.0, 0.0))#instantiate average vector
        AverageVector=AverageVector+FrustumInstance.LowerLeft
        AverageVector=AverageVector+FrustumInstance.LowerRight
        AverageVector=AverageVector+FrustumInstance.UpperRight
        AverageVector=AverageVector+FrustumInstance.UpperLeft
        #persist into frustrum object the average point of the rectangular face
        FrustumInstance.AverageFrontFacePoint=AverageVector/4
        
        
        
    #now get the lengths of the triangle which makes up the horizontal (so origin of frustrum, lowerleft and lowerright)
    for FrustumInstance in FrustrumList:
        FrustumInstance.Triangle_Horiz_Left_length=DistanceBtwn_Vectors(FrustumInstance.OriginPosition,FrustumInstance.LowerLeft)
        FrustumInstance.Triangle_Horiz_Right_Length=DistanceBtwn_Vectors(FrustumInstance.OriginPosition,FrustumInstance.LowerRight)
        FrustumInstance.Triangle_Horiz_Front_Length=DistanceBtwn_Vectors(FrustumInstance.LowerLeft,FrustumInstance.LowerRight)
        
    #get a b c angles of triangle
    for FrustumInstance in FrustrumList:
        #use law of cosines rule to get angles of triangle
        a=FrustumInstance.Triangle_Horiz_Left_length
        b=FrustumInstance.Triangle_Horiz_Right_Length
        c=FrustumInstance.Triangle_Horiz_Front_Length
        angA = angle(a,b,c)
        angB = angle(b,c,a)
        angC = angle(c,a,b)
        FrustumInstance.Triangle_Horiz_FOV_Deg=angA
        #angB and C are the big angles at rectangular face of frustrum 
        #angleA is angular FOV
        #step through interpretation of triangles
        step1=90-angC#assume another triangle in shadow of frustrum triangle forming a square
        step2=180-90-step1#this is angle of backplane of frustrum (where origin is)
        step3=(angA/2)+step2#if there is no slew in frustrum - this should be 90 degrees
        #angles b and c are on the face of the frustrum
        FrustumInstance.Triangle_Horiz_AngleDiff=(step3-90)#get difference from central position
      
      
      
    #now get the lengths of the triangle which makes up the vertical (so origin of frustrum, upperleft and lowerleft)
    for FrustumInstance in FrustrumList:
        FrustumInstance.Triangle_Vert_Left_length=DistanceBtwn_Vectors(FrustumInstance.OriginPosition,FrustumInstance.LowerLeft)
        FrustumInstance.Triangle_Vert_Right_Length=DistanceBtwn_Vectors(FrustumInstance.OriginPosition,FrustumInstance.UpperLeft)
        FrustumInstance.Triangle_Vert_Front_Length=DistanceBtwn_Vectors(FrustumInstance.LowerLeft,FrustumInstance.UpperLeft)
    #get a b c angles of triangle
    for FrustumInstance in FrustrumList:
        #use law of cosines rule to get angles of triangle
        a=FrustumInstance.Triangle_Vert_Left_length
        b=FrustumInstance.Triangle_Vert_Right_Length
        c=FrustumInstance.Triangle_Vert_Front_Length
        angA = angle(a,b,c)
        angB = angle(b,c,a)
        angC = angle(c,a,b)
        FrustumInstance.Triangle_Vert_FOV_Deg=angA
        #angB and C are the big angles at rectangular face of frustrum
        #angleA is angular FOV
        #step through interpretation of triangles
        step1=90-angC#assume another triangle in shadow of frustrum triangle forming a square
        step2=180-90-step1#this is angle of backplane of frustrum (where origin is)
        step3=(angA/2)+step2#if there is no slew in frustrum - this should be 90 degrees
        #angles b and c are on the face of the frustrum
        FrustumInstance.Triangle_Vert_AngleDiff=(step3-90)#get difference from central position

        
        #debug code to print origin of frustrum and associated camera ID
#        bpy.ops.object.empty_add(radius=1, location=(FrustrumList[indexer].OriginPosition))
#        #print camera name to debug if frustrum averages match index
#        bpy.ops.object.text_add(radius=0.4,enter_editmode=False, align='WORLD', location=(FrustrumList[indexer].OriginPosition),rotation=(90.0, 0.0, 180.0))
#        #select last object (text object) and label the landmark with the dlib index
#        ob=bpy.context.object
#        ob.data.body = str(CameraObjectID)
        
    #now should have a fully populated frustrum data object
    for FrustumInstance in FrustrumList:
        print(FrustumInstance.Name, FrustumInstance.Triangle_Horiz_FOV_Deg,FrustumInstance.Triangle_Vert_FOV_Deg)
        
    return FrustrumList

def Deltree(Folderpath):
      # check if folder exists
    if len(Folderpath)<10:
        raise("Input:" + str(Folderpath))
        raise ValueError("Deltree error - path too short warning might be root!")
        return
    if os.path.exists(Folderpath):
         # remove if exists
         shutil.rmtree(Folderpath)
    else:
         # throw your exception to handle this special scenario
         #raise Exception("Unknown Error trying to Deltree: " + Folderpath)
         pass
    return

def DeleteFiles_RecreateFolder(FolderPath):
    Deltree(FolderPath)
    os.mkdir(FolderPath)


 