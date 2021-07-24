import os
import argparse


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    assert os.path.exists(inputdir), 'Input dir not found'
    assert os.path.exists(targetdir), 'target dir not found'
    mkdir(outputdir)
    vids = os.listdir(inputdir)
    for vid in vids:

        imgs = os.listdir(os.path.join(inputdir, vid))
        imgs= sorted(imgs)
        length = len(imgs)

        for idx in range(length):
            groups = ''
            if idx == 0:
                groups += os.path.join(inputdir, vid,  imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid,  imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid,  imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid,  imgs[idx+1]) + '|'
                groups += os.path.join(inputdir, vid,  imgs[idx+2]) + '|'
                groups += os.path.join(targetdir, vid, imgs[idx])
            elif idx == 1:
                groups += os.path.join(inputdir, vid, imgs[idx-1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx-1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx+1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx+2]) + '|'
                groups += os.path.join(targetdir, vid, imgs[idx])
            elif idx == length-2:
                groups += os.path.join(inputdir, vid, imgs[idx-2]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx-1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx+1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx+1]) + '|'
                groups += os.path.join(targetdir, vid, imgs[idx])
            elif idx == length-1:
                groups += os.path.join(inputdir, vid, imgs[idx-2]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx-1]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx]) + '|'
                groups += os.path.join(inputdir, vid, imgs[idx]) + '|'
                groups += os.path.join(targetdir, vid, imgs[idx])
            else:
                for i in range(idx-2, idx+3):
                    groups += os.path.join(inputdir, vid, imgs[i]) + '|'
                groups += os.path.join(targetdir, vid, imgs[idx])
            with open(os.path.join(outputdir, 'groups.txt'), 'a') as f:
                f.write(groups + '\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--input', type=str, default='/gdata1/zhuqi/REDS/train/train_blur', metavar='PATH', help='root dir to save low resolution images')
    parser.add_argument('--target', type=str, default='/gdata1/zhuqi/REDS/train/train_sharp', metavar='PATH', help='root dir to save high resolution images')
    parser.add_argument('--output', type=str, default='/gdata1/zhuqi/REDS/train', metavar='PATH', help='output dir to save group txt files')
    parser.add_argument('--ext', type=str, default='.png', help='Extension of files')
    args = parser.parse_args()

    inputdir = args.input
    targetdir = args.target
    outputdir = args.output
    ext = args.ext

    main()