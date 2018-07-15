#!/usr/bin/python
import os, sys, math

#baseCom = "screen -dmS %s /bin/nice -n 10 /usr/local/lib/matlab/bin/matlab -nosplash -nodesktop -nojvm -singleCompThread"
baseCom = "screen -dmS %s /usr/bin/nice -n 10 /home/aayush/MATLAB/bin/matlab -nosplash -nodesktop -nojvm"

count = int(sys.argv[1])
command = sys.argv[2]
name = sys.argv[3]

numDigitsFormat = int(math.ceil(math.log(count) / math.log(10)))
nameStr = "%s%%0%dd" % (name, numDigitsFormat)
#get the 

for i in range(count):
    instanceName = nameStr % i
    instanceCom = (baseCom % instanceName) + " -r "+  command; # + "(" + str(i+1) + ")" + '"'
    print "Starting instance %d/%d" % (i+1,count)
    print instanceCom
    os.system(instanceCom)


