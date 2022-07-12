clc; clear; close all; warning off all;

%%% Proses pelatihan
% menetapkan data latih
nama_folder = 'data training';
% membaca file dengan ekstensi .jpg
nama_file = dir (fullfile(nama_folder,'*.jpg'));
% membaca jumlah file yang berekstensi .jpg
jumlah_file = numel(nama_file);
% menginisialisasi variabel ciri latih
ciri_latih = zeros(jumlah_file,4);
% melakukan pengolahan citra terhadap seluruh file
for n = 1:jumlah_file
    % membaca file citra RGB
    Img = imread(fullfile(nama_folder,nama_file(n).name));
   % figure, imshow(Img)
    % melakukan konversi citra RGB menjadi citra grayscale
    Img_gray = rgb2gray(Img);
   % figure, imshow(Img_gray)
    % melakukan konversi citra grayscale menjadi citra biner
    bw = imbinarize(Img_gray);
   % figure, imshow(bw)
    % melakukan operasi komplemen
    bw = imcomplement(bw);
   % figure, imshow(bw)
    % melakukan operasi morfologi filling holes
    bw = imfill(bw,'holes');
   % figure, imshow(bw)
    % ekstraksi ciri
    % melakukan konversi citra rgb menjadi citra HSV
    HSV = rgb2hsv(Img);
   % figure, imshow(HSV)
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
    ciri_latih(n,1) = Hue;
    ciri_latih(n,2) = Saturation;
    ciri_latih(n,3) = Value;
    ciri_latih(n,4) = Luas;
    
end
  % menyusun variable kelas_latih
    kelas_latih = cell(jumlah_file,1);
    % mengisi nama-nama jenis aglaonema pada variable kelas latih
    for k = 1:8
        kelas_latih{k} = 'butterfly';
    end
    for k = 9:14
        kelas_latih{k} = 'moonlight';
    end
    for k = 15:19
        kelas_latih{k} = 'peacock';
    end
    
    % klasifikasi citra menggunakan algoritma naive bayes
    Mdl = fitcnb(ciri_latih,kelas_latih);
    
    % membaca kelas keluaran hasil pelatihan
    hasil_latih = predict(Mdl,ciri_latih);
    % menghitung akurasi pelatihan
    jumlah_benar = 0;
    for k = 1:jumlah_file
        if isequal(hasil_latih{k}, kelas_latih{k})
            jumlah_benar = jumlah_benar+1;
        end
    end

    akurasi_pelatihan = jumlah_benar/jumlah_file*100
    
    % menyimpan model naive bayes hasil pelatihan
    save Mdl Mdl



  