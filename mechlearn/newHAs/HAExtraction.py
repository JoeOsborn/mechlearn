import sys

if len(sys.argv) < 3:
  print('Use: "python HAExtraction.py [file with list of relevant files] [name to save]"')
  exit(0)

allHAs = sys.argv[1]
newFile = sys.argv[2]

files = [line.rstrip('\n') for line in open(allHAs)]
propsList = ['name']

for f in files:
  with open(f) as infile:
    for line in infile:
      if 'Warning' not in line:
        line = ("".join(line.rstrip().split())).split(':')
        if (line[1]) not in propsList:
          propsList.append(line[1])

with open(newFile, 'w') as file:

  for p in propsList:
    file.write(p)
    file.write('\t')
  file.write('\n')

  for f in files:
    t = {}
    with open(f) as infile:
      t['name'] = f
      for line in infile:
        if 'Warning' in line:
          continue
        else:
          line = line.rstrip().split(':')
          t["".join(line[1].split())] = line[2]
    for p in propsList:
      if p in t:
        file.write(t[p])
      else:
        file.write('None')
      file.write('\t')
    file.write('\n')