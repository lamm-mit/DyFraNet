from PIL import Image
import glob
import argparse

parser = argparse.ArgumentParser(description='img2gif')
parser.add_argument('loc', nargs='?', type=str)
parser.add_argument('end', nargs='?', type=str, default='png')
parser.add_argument('duration', nargs='?', type=int, default=100)
parser.add_argument('mode', nargs='?', type=str, default='bw')

args = parser.parse_args()
print('in',args.loc,'end',args.end,'duration',args.duration)
mode=args.mode
# Create the frames
frames = []
imgs = sorted(glob.glob(args.loc+'/*.'+args.end), key=lambda x: (len(x), x))

print('found',len(imgs),'images.',imgs)
for i in imgs:
    if mode == 'bw':
        new_frame = Image.open(i)#.convert('RGB').convert('L')
    else:
        new_frame = Image.open(i)#.convert('RGB').convert('L')
    frames.append(new_frame)

# Save into a GIF file that loops forever
if args.loc == '.':
    args.loc = 'result'
frames[0].save(args.loc+'.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=args.duration, loop=0)
