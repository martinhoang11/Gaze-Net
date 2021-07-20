import os
import sys
import numpy as np
import data_processing_core as dpc
import conversion as co
import cv2
from scipy import io as sio
import argparse


def read_file(path):
    with open(path) as infile:
        lines = infile.readlines()
        lines.pop(0)
        acc = lines.pop(-1)

    for line in lines:
        line = line.strip().split(" ")
        
        gt = line[2]
        name = line[0]
        prediction = line[1]
        
        gt = np.array(list(map(eval, gt.split(","))))
        prediction = np.array(list(map(eval, prediction.split(","))))

        yield name, gt, prediction

def convert_mpii_origin(screen_pose, logfile, annofolder, newannofolder, gtnum, is3D):
    # Read Screen-Camera pose
    screen_pose = sio.loadmat(screen_pose)
    rvec = screen_pose["rvects"]
    tvec = screen_pose["tvecs"]
    rmat = cv2.Rodrigues(rvec)[0]

    # read gaze origin from annotation file
    person = os.path.split(logfile)[0]
    person = os.path.split(person)[1].split(".")[0]

    annotation = os.path.join(annofolder, f"{person}.label")
    with open(annotation) as infile:
        lines = infile.readlines()
        lines.pop(0)

    newannotation = os.path.join(newannofolder, f"{person}.label")
    with open(newannotation) as infile:
        newannos = infile.readlines()
        newannos.pop(0)
         
    total_nloss = 0
    total_oloss = 0
    for count, (name, gt, prediction) in enumerate(read_file(logfile)):

        # Get the prediction in normalized space.
        annos = lines[count].strip().split(" ")
        if not is3D:
            prediction = dpc.GazeTo3d(prediction)
            gt = dpc.GazeTo3d(gt)
    

        # Convert normalized space into CCS.
        Rvec = np.array(list(map(eval, annos[-3].split(","))))
        Svec = np.array(list(map(eval, annos[-2].split(","))))
        origin = np.array(list(map(eval, annos[-1].split(","))))

        Rmat = cv2.Rodrigues(Rvec)[0]
        Smat = np.diag(Svec)
        mat = np.dot(np.linalg.inv(Rmat), np.linalg.inv(Smat))

        pred_ccs = np.dot(mat, prediction.reshape((3, 1))).flatten()
        origin_ccs = np.dot(mat, origin.reshape((3,1))).flatten()


        # Get the info of new origin.
        anno2 = newannos[count].strip().split(" ")
        gt2 = np.array(list(map(eval, anno2[gtnum].split(","))))

        Rvec2 = np.array(list(map(eval, anno2[-3].split(","))))
        Svec2 = np.array(list(map(eval, anno2[-2].split(","))))
        origin2 = np.array(list(map(eval, anno2[-1].split(","))))

        Rmat2 = cv2.Rodrigues(Rvec2)[0]
        Smat2 = np.diag(Svec2)
        mat2 = np.dot(np.linalg.inv(Rmat2), np.linalg.inv(Smat2))

        origin2_ccs = np.dot(mat2, origin2.reshape((3,1)))

        # Compute the new direction.
        gaze2 = co.ToNewOrigin(pred_ccs, origin_ccs, origin2_ccs, rmat, tvec)
        gaze2 = np.dot(Smat2, np.dot(Rmat2, gaze2.reshape(3,1)))

        loss = dpc.AngularLoss(gaze2.flatten(), gt2.flatten())
        total_nloss += loss

        loss = dpc.AngularLoss(prediction.flatten(), gt.flatten())
        total_oloss += loss

    return total_nloss, total_oloss, count+1


def main(args):

    total_nloss = 0
    total_oloss = 0
    total_count = 0

    for i in ["p00", "p01", "p02", "p03", "p04", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13", "p14"]:

        logfile =  os.path.join(args.logfolder, f"{i}.label/{args.name}")

        nloss, oloss, count= convert_mpii_origin(args.screen, logfile, args.sourcelabel, args.targetlabel, args.gtnum, args.is3D)

        total_nloss += nloss
        total_oloss += oloss
        total_count += count
    return total_nloss/total_count, total_oloss/total_count, total_count
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert origin on MPIIGaze")


    parser.add_argument('--screen', type=str, default = "./MPIIFaceGaze/p00/Calibration/monitorPose.mat", help="The path of calibration matrix. Need not be changed") 

    parser.add_argument('-f', '--logfolder', type=str, default = "./evaluation/", help="The folder saving result file") 

    parser.add_argument('-n', '--name', type=str, default = "20.log", help="The specifc filename in the logfoder.") 

    parser.add_argument('-s', '--sourcelabel', type=str, default = "./MPIIFaceGaze/Label/", help="label file of the log") 

    parser.add_argument('-t', '--targetlabel', type=str, default = "./MPIIGaze/Label/", help="label file of the new origin") 

    parser.add_argument('-g', '--gtnum', type=int, default = 3, help="label file of the new origin")
    
    parser.add_argument('-3d', '--is3D', type=int, default = 0, help="3D label") 

    args = parser.parse_args()
    acc, count = main(args)
    print("==>Angluar Loss: ", end="")
    print(acc)



  
          
          
      


