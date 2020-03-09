
import os
import glob


def main():
    folder = '/data/zming/datasets/Hand/hand_frames'
    files = glob.glob(os.path.join(folder, '*.png'))
    for file in files:
        strtmp = str.split(file,'/')[-1]
        strtmp = str.split(strtmp, '_')
        frmnum = int(strtmp[0][5:])
        os.rename(file, os.path.join(folder, 'frame%04d_%s'%(frmnum,strtmp[1])))




















if __name__ == '__main__':
    main()
