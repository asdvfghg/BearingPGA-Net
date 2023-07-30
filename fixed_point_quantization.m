clear all
% Select input and output folder
input_folder = 'Weight_Parameters/Float';
output_folder = 'Weight_Parameters/Fixed_Point';

file_names = dir(fullfile(input_folder, '*.txt'));

for i = 1:length(file_names)
    file_path = fullfile(input_folder, file_names(i).name);
    data = importdata(file_path);
    data_max = max(data);
    data_min = min(data);
    % determin fractionlength
    if i == 1
        if (data_max<1)&&(data_min>-1)
            fractionlength = 14;
        elseif (data_max<3)&&(data_min>-3)
            fractionlength = 13;  
        elseif (data_max<7)&&(data_min>-7)
            fractionlength = 12;
        elseif (data_max<15)&&(data_min>-15)
            fractionlength = 11;
        elseif (data_max<31)&&(data_min>-31)
            fractionlength = 10;
        end
            temp = fractionlength;
    end
    % keep the same bit-width of CONV weight and bias
    if i == 2
        fractionlength = temp;
    end
    % The integers bit-width of FC should be set larger to align the output
    if (i == 3) || (i == 4)
        fractionlength = 8;
    end
    disp(fractionlength);
    % quantization
    data_fixed_q = quantizer('fixed','round','saturate',[16 fractionlength]);
    % convert to binary
    data_q = num2bin(data_fixed_q, data);
    output_file_path = fullfile(output_folder, file_names(i).name);
    dlmwrite(output_file_path, data_q, 'delimiter', '');
end
%%
% Note: Matlab generate the complement numbers, which should be converted to original numbers
% via convert_complement_number_to_original.cpp
