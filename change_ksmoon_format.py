
totalFace= 24384

ksmoon_file = open("/home/bkko/ml_study/aflw/aflw/annotation/annot_ksmoon", 'r')
new_file = open("/home/bkko/ml_study/aflw/aflw/annotation/annot_bkko", 'w')

ksmoon_array = ksmoon_file.readlines()

img_path = 0
left = 0
btm = 0
width = 0
height = 0
face_cnt = 0
face_annot = []

i = 0
for k in xrange(totalFace) :
	face_annot = []
	face_cnt = 1

	ksmoon_line = ksmoon_array[i].rstrip()

	img_path, left, btm, width, heigth = ksmoon_line.split(',')
	face_annot = [ [left, btm, width, height] ]

	new_file.write("%s\n" %img_path)
	j = i + 1
	while 1 :
		if j == totalFace :
			break
		if ksmoon_array[j].rstrip().split(',')[0] != img_path :
			break
		face_cnt = face_cnt + 1
		face_annot.append( ksmoon_array[j].rstrip().split(',')[1:5] )
		j = j + 1

	new_file.write("%s\n" %face_cnt )
	for j in xrange(len(face_annot) ) :
		new_file.write("%s %s %s %s\n" %(face_annot[j][0], face_annot[j][1], face_annot[j][2], face_annot[j][2] ) )

	i = i + face_cnt

