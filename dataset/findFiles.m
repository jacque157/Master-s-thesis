function [files] = findFiles(path)
    files = [];
    d = dir('foldername');
    dfolders = d([d(:).isdir]);
    dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));
    %for folder=folders
        %if isfolder(folder)
            %files = [files, findFiles(folder)];
        %else
            %files = [files, folder];
        %end
    %end
end
