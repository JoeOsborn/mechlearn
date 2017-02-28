import sys
import re

for f in sys.argv[1:]:
    with open(f) as infile:
        for line in infile:
            line = line.rstrip()
            uc_gravity = re.search('up\-control\-gravity\s\:\s([\-\d\.]+)',line)
            uf_gravity = re.search('up\-fixed\-gravity\s\:\s([\-\d\.]+)',line)
            gravity = re.search('gravity\s\:\s([\-\d\.]+)',line)
            maxDuration = re.search('maxButtonDuration\s\:\s([\-\d\.]+)',line)
            minDuration = re.search('minButtonDuration\s\:\s([\-\d\.]+)',line)
            upControlToUpFixedDYReset = re.search('upControlToUpFixedDYReset\s\:\s(.+)',line)    
            groundToUpControlDYReset = re.search('groundToUpControlDYReset\s\:\s(.+)',line)    
    
            if uc_gravity:
                print 'uc_gravity:', uc_gravity.group(1)
            elif uf_gravity:
                print 'uf_gravity:', uf_gravity.group(1)
            elif gravity:
                print 'gravity:', gravity.group(1)
            elif maxDuration:
                print 'maxDuration:', maxDuration.group(1)
            elif minDuration:
                print 'minDuration:', minDuration.group(1)
            elif upControlToUpFixedDYReset:
                fixed =  re.search('\(\'\+\',\s*([\d\.\-]+), \(\'\+',upControlToUpFixedDYReset.group(0))
                if fixed:
                    print 'upControl:',fixed.group(1)
                nonFixed = re.search('\(\'\*\',\s*([\d\.\-]+), \(\'y\'',upControlToUpFixedDYReset.group(0))
                if nonFixed:
                    print 'upControl:',nonFixed.group(1)
            elif groundToUpControlDYReset:
                fixed =  re.search('\(\'\+\',\s*([\d\.\-]+), \(\'\+',groundToUpControlDYReset.group(0))
                if fixed:
                    print 'upControlReset:',fixed.group(1)
                
