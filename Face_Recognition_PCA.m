% Author: Pyry Helkkula
% Special thanks to Dr Tae-Kyun Kim for providing the data set containing
% 520 face images of 52 persons
clear all
close all
clc
load face.mat
%% 1.Image data partitioning and organizing into field arrays
% The image data is organized by training and testing data and also by person

Faces.train = cat(2,X(:,1:8),X(:,11:18),X(:,21:28),X(:,31:38),X(:,41:48),X(:,51:58),X(:,61:68),X(:,71:78),X(:,81:88),X(:,91:98),X(:,101:108),X(:,111:118),X(:,121:128),X(:,131:138),X(:,141:148),X(:,151:158), ...
    X(:,161:168),X(:,171:178),X(:,181:188),X(:,191:198),X(:,201:208),X(:,211:218),X(:,221:228),X(:,231:238),X(:,241:248),X(:,251:258),X(:,261:268),X(:,271:278),X(:,281:288),X(:,291:298),X(:,301:308),X(:,311:318),X(:,321:328),X(:,331:338),X(:,341:348),X(:,351:358), ...
    X(:,361:368),X(:,371:378),X(:,381:388),X(:,391:398),X(:,401:408),X(:,411:418),X(:,421:428),X(:,431:438),X(:,441:448),X(:,451:458),X(:,461:468),X(:,471:478),X(:,481:488),X(:,491:498),X(:,501:508),X(:,511:518));

Faces.test = cat(2,X(:,9:10),X(:,19:20),X(:,29:30),X(:,39:40),X(:,49:50),X(:,59:60),X(:,69:70),X(:,79:80),X(:,89:90),X(:,99:100),X(:,109:110),X(:,119:120),X(:,129:130),X(:,139:140),X(:,149:150),X(:,159:160), ...
    X(:,169:170),X(:,179:180),X(:,189:190),X(:,199:200),X(:,209:210),X(:,219:220),X(:,229:230),X(:,239:240),X(:,249:250),X(:,259:260),X(:,269:270),X(:,279:280),X(:,289:290),X(:,299:300),X(:,309:310),X(:,319:320),X(:,329:330),X(:,339:340),X(:,349:350),X(:,359:360), ...
    X(:,369:370),X(:,379:380),X(:,389:390),X(:,399:400),X(:,409:410),X(:,419:420),X(:,429:430),X(:,439:440),X(:,449:450),X(:,459:460),X(:,469:470),X(:,479:480),X(:,489:490),X(:,499:500),X(:,509:510),X(:,519:520));

%% 2. PCA on training data by directly computing eigenvectors and eigenvalues of S data covariance matrix S
%clearvars X l % saving memory
N=length(Faces.train(1,:)); % number of images
Xmean=mean(Faces.train,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(Faces.train(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
tic
S=(1/N)*(Xprime*Xprime'); % S: covariance matrix
[eigVec,eigVal]=eig(S); % Finding eigenvectors and values to build eigen-subspaces, NB: S*U=lambda*U (Lec 13-14 slide 19)
elapsedtime=toc
[sortedValues,sortIndex] = sort(diag(eigVal),'descend');
LargestEigenValues416 = sortedValues(1:N);
LargestEigenVectors416 = eigVec(:,sortIndex(1:N));

%% 3.
% Ratio of Variance
VARtot=trace(eigVal);
[sortedValues,~] = sort(diag(eigVal),'descend');
%Ratio of Variance
VARratio=zeros(1,length(S));
for D=1:length(S)
    VARratio(D)=sum(sortedValues(1:D))/VARtot;
end
figure
plot(VARratio,'LineWidth', 2.5)
set(gca,'FontSize',20)
title('Variance Ratio of Data Covariance Matrix S')
xlabel('Number of Principal Components')
ylabel('Variance Ratio')

% Mean Image
Imean=reshape(Xmean(:,1),56,46);
figure
imagesc(Imean)
set(gca,'FontSize',20)
title('Mean Image of Training Data')
colormap(gray)

%% 4. PCA on training data 
N=length(Faces.train(1,:)); % number of images
Xmean=mean(Faces.train,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(Faces.train(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
tic
S_=(1/N)*(Xprime'*Xprime); % data covariance matrix
[eigVec_,eigVal_]=eig(S_); % Finding lambda_m and u_m as in Q1
U416=Xprime*eigVec_; %matrix of 416 largest eigenvectors
U416=normc(U416); % normalize U416
elapsedtime=toc
Eigenvalues=diag(eigVal_);

%% 5.
N=length(Faces.train(1,:)); % number of images
Xmean=mean(Faces.train,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(Faces.train(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
S_=(1/N)*(Xprime'*Xprime); % data covariance matrix
[eigVec_,eigVal_]=eig(S_); % Finding lambda_m and u_m as in Q1
U416=Xprime*eigVec_; %matrix of 416 largest eigenvectors
U416=normc(U416); % normalize U416
%% Reconstruction of example image with varying number of bases
% Test image 1: Person 1, 1st training Image
Faces.train=struct('Person1',{X(:,1:8)}); % create field for Person1 training data
D=length(Faces.train.Person1(:,1));
X1=Faces.train.Person1(:,1);
X1=reshape(X1,56,46);
figure
subplot(2,3,1)
imagesc(X1)
set(gca,'FontSize',20)
title('Original Image')
colormap(gray)
%%%% 1 base
X1=Faces.train.Person1(:,1);
X1mean=mean(X1);
X1mean=repmat(X1mean,[D 1]);
X1prime=(X1-X1mean);
S1=X1prime*X1prime';
[eigVec1,eigVal1]=eig(S1);
[~,sortIndex] = sort(diag(eigVal1),'descend');
maxIndex = sortIndex(1:1);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec1(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA1=U416*eigVec';
PCA1mean=mean(PCA1(:,maxIndex),2);
PCA1mean=reshape(PCA1mean,56,46);
subplot(2,3,2)
imagesc(PCA1mean)
set(gca,'FontSize',20)
title('1 base')
colormap(gray)
%%%%% 5 bases
X1mean=mean(X1);
X1mean=repmat(X1mean,[D 5]);
X1=repmat(X1,[1 5]);
X1prime=(X1-X1mean);
S1=X1prime*X1prime';
[eigVec1,eigVal1]=eig(S1);
[~,sortIndex] = sort(diag(eigVal1),'descend');
maxIndex = sortIndex(1:5);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec1(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA1=U416*eigVec';
PCA1mean=mean(PCA1(:,maxIndex),2);
PCA1mean=reshape(PCA1mean,56,46);
subplot(2,3,3)
imagesc(PCA1mean)
set(gca,'FontSize',20)
title('5 bases')
colormap(gray)
%%%% 10 bases
X1mean=mean(X1);
X1mean=repmat(X1mean,[D 10]);
X1=repmat(X1,[1 10]);
X1prime=(X1-X1mean);
S1=X1prime*X1prime';
[eigVec1,eigVal1]=eig(S1);
[~,sortIndex] = sort(diag(eigVal1),'descend');
maxIndex = sortIndex(1:10);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec1(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA1=U416*eigVec';
PCA1mean=mean(PCA1(:,maxIndex),2);
PCA1mean=reshape(PCA1mean,56,46);
subplot(2,3,4)
imagesc(PCA1mean)
set(gca,'FontSize',20)
title('10 bases')
colormap(gray)
%%%%% 20 bases
X1mean=mean(X1);
X1mean=repmat(X1mean,[D 20]);
X1=repmat(X1,[1 20]);
X1prime=(X1-X1mean);
S1=X1prime*X1prime';
[eigVec1,eigVal1]=eig(S1);
[~,sortIndex] = sort(diag(eigVal1),'descend');
maxIndex = sortIndex(1:20);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec1(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA1=U416*eigVec';
PCA1mean=mean(PCA1(:,maxIndex),2);
PCA1mean=reshape(PCA1mean,56,46);
subplot(2,3,5)
imagesc(PCA1mean)
set(gca,'FontSize',20)
title('20 bases')
colormap(gray)
%%%%% 30 bases
X1mean=mean(X1);
X1mean=repmat(X1mean,[D 30]);
X1=repmat(X1,[1 30]);
X1prime=(X1-X1mean);
S1=X1prime*X1prime';
[eigVec1,eigVal1]=eig(S1);
[~,sortIndex] = sort(diag(eigVal1),'descend');
maxIndex = sortIndex(1:30);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec1(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA1=U416*eigVec';
PCA1mean=mean(PCA1(:,maxIndex),2);
PCA1mean=reshape(PCA1mean,56,46);
subplot(2,3,6)
imagesc(PCA1mean)
set(gca,'FontSize',20)
title('30 bases')
colormap(gray)