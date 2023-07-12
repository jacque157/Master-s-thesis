function [pcls] = readDat(fname)
    fid = fopen(fname,'rb');
    ar = fread(fid, 3, 'uint32=>uint32');
    no_scans = ar(1)
    x_res = ar(2)
    y_res = ar(3)
    pcls = cell(no_scans, 1);
    for i=1:no_scans
        scan = fread(fid, x_res*y_res*3, 'float32=>float32');
        scan = double(reshape(scan, 3, 344*257))';
        pcls(i) = {scan};
        pcshow(scan);
        pause;
    end
    fclose(fid);

%     fid = fopen(fname, 'rb');
%     fseek(fid, 2*512*424*(idx-1), 'bof');
%     data1 = fread(fid, 512*424, 'uint16=>uint16');
%     fclose(fid);
%     % Data is stored in row-major.
%     im = double(reshape(data1, 512, 424))';
%     im = im(:,end:-1:1,:); % Flip left-right
end  