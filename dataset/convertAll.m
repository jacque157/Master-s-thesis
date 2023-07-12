%paths = [string('CMU/male/Part 2/'), string('CMU/male/Part 3/'), string('CMU/male/Part 4/'), string('CMU/male/Part 5/'), string('CMU/male/Part 6/'), string('CMU/female/Part 1/'), string('CMU/female/Part 2/')];
%paths = [string('ACCAD/male/Part 1/'), string('ACCAD/male/Part 2/'), string('ACCAD/female/')];
%paths = [string('EKUT/male/')];
%paths = [string('Eyes_Japan/male/')];

for path = paths
    convert2mat(path);   
end