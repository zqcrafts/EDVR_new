import glob
import os
import sys
sys.path.append('/code/EDVR_ORI')
import argparse

# put this file into your video path

def mkdir(dirs):
	if not os.path.exists(dirs):
		os.mkdir(dirs)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, default='/output/video', help='Path to store cideo files')
	args = parser.parse_args()
	videos = os.listdir('/output/Results')
	mkdir(args.output)
	for vid in videos:
		# mkdir(os.path.join(args.output, vid))
		# cmd = 'ffmpeg -f image2 -i {}/%05d.jpg -vcodec libx265 {}/{}.mp4'.format(vid, args.output, vid)
		cmd = 'ffmpeg -r 24000/1001 -i {}/%5d.png -vcodec libx265 -pix_fmt yuv422p -crf 10 {}/{}.mp4'.format('/output/Results/'+vid,
																											args.output, vid)
		os.system(cmd)
