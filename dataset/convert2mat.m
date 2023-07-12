function [] = convert2mat(path)
    mat_folder = strcat('mat/', path);
    mkdir(mat_folder);
    files = dir(path);
    
    for i =1 : size(files)
        file = files(i).name;
        if strcmp(file, '.') || strcmp(file, '..') || endsWith(file, '.obj')
            continue
        end
    
        in_path = strcat(path, file);
        fid = fopen(in_path,'rb');
        ar = fread(fid, 3, 'uint32=>uint32');
        
        no_scans = ar(1);
        x_res = ar(2);
        y_res = ar(3);
    
        sep1 = strfind(file, '.');
        sequence = file(1:sep1-1);
    
        for i=1:no_scans
            scan = fread(fid, x_res*y_res*3, 'float32=>float32');
            scan = double(reshape(scan, 3, x_res, y_res));
            x = reshape(scan(1, :, :), x_res, y_res)';
            y = reshape(scan(2, :, :), x_res, y_res)';
            z = reshape(scan(3, :, :), x_res, y_res)';
            img = cat(3, x, y, z);
            out_path = strcat(mat_folder, sequence, '_point_cloud_', string(i), '.mat');
            save(out_path, "img");
        end
        fclose(fid);
    end
end

