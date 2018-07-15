#!/usr/bin/python
import os, sys, math

count = int(sys.argv[1])
name = sys.argv[2]

numDigitsFormat = int(math.ceil(math.log(count) / math.log(10)))
nameStr = "%s%%0%dd" % (name, numDigitsFormat)
#get the 

for i in range(count):
    instanceName = nameStr % i
    os.system("screen -S %s -X quit" % instanceName)

