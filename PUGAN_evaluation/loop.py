import os

DIR=os.listdir('/home/steven/PU-GAN_data/test_off/')
print(DIR)
names = [x.split('.')[0] for x in DIR]
print(names)
for i in names:
	os.system('./evaluation '+'../../PU-GAN_data/test_off/'+i+'.off '+'../outputs/t3_50_upsample/'+i+'.xyz')