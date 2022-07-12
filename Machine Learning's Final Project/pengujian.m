clc; clear; close all; warning off all;

%%% Proses pengujian
% menetapkan data uji
nama_folder = 'data uji';
% membaca file dengan ekstensi .jpg
nama_file = dir (fullfile(nama_folder,'*.jpg'));
% membaca jumlah file yang berekstensi .jpg
jumlah_file = numel(nama_file);

% menginisialisasi variabel ciri_uji
ciri_uji = zeros(jumlah_file,4);
% melakukan pengolahan citra terhadap seluruh file
for n = 1:jumlah_file
    % membaca file citra RGB
    Img = imread(fullfile(nama_folder,nama_file(n).name));
 %   figure, imshow(Img)
    % melakukan konversi citra RGB menjadi citra grayscale
    Img_gray = rgb2gray(Img);
 %   figure, imshow(Img_gray)
    % melakukan konversi citra grayscale menjadi citra biner
    bw = imbinarize(Img_gray);
 %   figure, imshow(bw)
    % melakukan operasi komplemen
    bw = imcomplement(bw);
 %   figure, imshow(bw)
    % melakukan operasi morfologi filling holes
    bw = imfill(bw,'holes');
%    figure, imshow(bw)
    % ekstraksi ciri
    % melakukan konversi citra rgb menjadi citra HSV
    HSV = rgb2hsv(Img);
%    figure, imshow(HSV)
    % mengekstrak komponen h, s, v pada citra hsv
    H = HSV(:,:,1); %Hue
    S = HSV(:,:,2); %Saturation
    V = HSV(:,:,3); %Value
    % mengubah nilai piksel background menjadi nol
    H(~bw) = 0;
    S(~bw) = 0;
    V(~bw) = 0;
    % menghitung nilai ratarata h,s,v
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    % menghitung luas objek
    Luas = sum(sum(bw));
    % mengisi variabel ciri latih dengan ciri hasil ekstraksi
    ciri_uji(n,1) = Hue;
    ciri_uji(n,2) = Saturation;
    ciri_uji(n,3) = Value;
    ciri_uji(n,4) = Luas;
end

% menyusun variable kelas_uji
kelas_uji = cell(jumlah_file,1);
% mengisi nama-nama jenis aglaonema pada variable kelas latih
for k = 1:4
    kelas_uji{k} = 'butterfly';
end
for k = 5:7
    kelas_uji{k} = 'moonlight';
end
for k = 8:11
    kelas_uji{k} = 'peacock';
end

% memanggil model naive bayes hasil pelatihan
load Mdl

% membaca kelas keluaran hasil pengujian
hasil_uji = predict(Mdl,ciri_uji);

% menghitung akurasi pengujian
jumlah_benar = 0;
for k = 1:jumlah_file
    if isequal(hasil_uji{k}, kelas_uji{k})
        jumlah_benar = jumlah_benar+1;
    end
end

akurasi_pengujian = jumlah_benar/jumlah_file*100

% menyimpan model naive bayes hasil pengujian
    save Mdl Mdl