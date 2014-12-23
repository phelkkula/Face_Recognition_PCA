% Author: Pyry Helkkula
% Special thanks to Dr Tae-Kyun Kim for providing the data set containing
% 520 face images of 52 persons
clear all
close all
clc
load face.mat
%% 1.Image data partitioning and organizing into field arrays
% The image data is organized by training and testing data and also by person

ImageData.Training=struct('Person1',{X(:,1:8)},'Person2',{X(:,11:18)},'Person3',{X(:,21:28)},'Person4',{X(:,31:38)},'Person5',{X(:,41:48)},'Person6',{X(:,51:58)},'Person7',{X(:,61:68)},'Person8',{X(:,71:78)},'Person9',{X(:,81:88)},'Person10',{X(:,91:98)}, ...
    'Person11',{X(:,101:108)},'Person12',{X(:,111:118)},'Person13',{X(:,121:128)},'Person14',{X(:,131:138)},'Person15',{X(:,141:148)},'Person16',{X(:,151:158)},'Person17',{X(:,161:168)},'Person18',{X(:,171:178)},'Person19',{X(:,181:188)},'Person20',{X(:,191:198)}, ...
    'Person21',{X(:,201:208)},'Person22',{X(:,211:218)},'Person23',{X(:,221:228)},'Person24',{X(:,231:238)},'Person25',{X(:,241:248)},'Person26',{X(:,251:258)},'Person27',{X(:,261:268)},'Person28',{X(:,271:278)},'Person29',{X(:,281:288)},'Person30',{X(:,291:298)}, ...
    'Person31',{X(:,301:308)},'Person32',{X(:,311:318)},'Person33',{X(:,321:328)},'Person34',{X(:,331:338)},'Person35',{X(:,341:348)},'Person36',{X(:,351:358)},'Person37',{X(:,361:368)},'Person38',{X(:,371:378)},'Person39',{X(:,381:388)},'Person40',{X(:,391:398)}, ...
    'Person41',{X(:,401:408)},'Person42',{X(:,411:418)},'Person43',{X(:,421:428)},'Person44',{X(:,431:438)},'Person45',{X(:,441:448)},'Person46',{X(:,451:458)},'Person47',{X(:,461:468)},'Person48',{X(:,471:478)},'Person49',{X(:,481:488)},'Person50',{X(:,491:498)}, ...
    'Person51',{X(:,501:508)},'Person52',{X(:,511:518)},'TrainingData',{cat(2,X(:,1:8),X(:,11:18),X(:,21:28),X(:,31:38),X(:,41:48),X(:,51:58),X(:,61:68),X(:,71:78),X(:,81:88),X(:,91:98),X(:,101:108),X(:,111:118),X(:,121:128),X(:,131:138),X(:,141:148),X(:,151:158), ...
    X(:,161:168),X(:,171:178),X(:,181:188),X(:,191:198),X(:,201:208),X(:,211:218),X(:,221:228),X(:,231:238),X(:,241:248),X(:,251:258),X(:,261:268),X(:,271:278),X(:,281:288),X(:,291:298),X(:,301:308),X(:,311:318),X(:,321:328),X(:,331:338),X(:,341:348),X(:,351:358), ...
    X(:,361:368),X(:,371:378),X(:,381:388),X(:,391:398),X(:,401:408),X(:,411:418),X(:,421:428),X(:,431:438),X(:,441:448),X(:,451:458),X(:,461:468),X(:,471:478),X(:,481:488),X(:,491:498),X(:,501:508),X(:,511:518))});

ImageData.Testing=struct('Person1',{X(:,9:10)},'Person2',{X(:,19:20)},'Person3',{X(:,29:30)},'Person4',{X(:,39:40)},'Person5',{X(:,49:50)},'Person6',{X(:,59:60)},'Person7',{X(:,69:70)},'Person8',{X(:,79:80)},'Person9',{X(:,89:90)},'Person10',{X(:,99:100)}, ...
    'Person11',{X(:,109:110)},'Person12',{X(:,119:120)},'Person13',{X(:,129:130)},'Person14',{X(:,139:140)},'Person15',{X(:,149:150)},'Person16',{X(:,159:160)},'Person17',{X(:,169:170)},'Person18',{X(:,179:180)},'Person19',{X(:,189:190)},'Person20',{X(:,199:200)}, ...
    'Person21',{X(:,209:210)},'Person22',{X(:,219:220)},'Person23',{X(:,229:230)},'Person24',{X(:,239:240)},'Person25',{X(:,249:250)},'Person26',{X(:,259:260)},'Person27',{X(:,269:270)},'Person28',{X(:,279:280)},'Person29',{X(:,289:290)},'Person30',{X(:,299:300)}, ...
    'Person31',{X(:,309:310)},'Person32',{X(:,319:320)},'Person33',{X(:,329:330)},'Person34',{X(:,339:340)},'Person35',{X(:,349:350)},'Person36',{X(:,359:360)},'Person37',{X(:,369:370)},'Person38',{X(:,379:380)},'Person39',{X(:,389:390)},'Person40',{X(:,399:400)}, ...
    'Person41',{X(:,409:410)},'Person42',{X(:,419:420)},'Person43',{X(:,429:430)},'Person44',{X(:,439:440)},'Person45',{X(:,449:450)},'Person46',{X(:,459:460)},'Person47',{X(:,469:470)},'Person48',{X(:,479:480)},'Person49',{X(:,489:490)},'Person50',{X(:,499:500)}, ...
    'Person51',{X(:,509:510)},'Person52',{X(:,519:520)},'TestingData',{cat(2,X(:,9:10),X(:,19:20),X(:,29:30),X(:,39:40),X(:,49:50),X(:,59:60),X(:,69:70),X(:,79:80),X(:,89:90),X(:,99:100),X(:,109:110),X(:,119:120),X(:,129:130),X(:,139:140),X(:,149:150),X(:,159:160), ...
    X(:,169:170),X(:,179:180),X(:,189:190),X(:,199:200),X(:,209:210),X(:,219:220),X(:,229:230),X(:,239:240),X(:,249:250),X(:,259:260),X(:,269:270),X(:,279:280),X(:,289:290),X(:,299:300),X(:,309:310),X(:,319:320),X(:,329:330),X(:,339:340),X(:,349:350),X(:,359:360), ...
    X(:,369:370),X(:,379:380),X(:,389:390),X(:,399:400),X(:,409:410),X(:,419:420),X(:,429:430),X(:,439:440),X(:,449:450),X(:,459:460),X(:,469:470),X(:,479:480),X(:,489:490),X(:,499:500),X(:,509:510),X(:,519:520))});

%% 2. PCA on training data by directly computing eigenvectors and eigenvalues of S data covariance matrix S
clearvars X l % saving memory
N=length(ImageData.Training.TrainingData(1,:)); % number of images
Xmean=mean(ImageData.Training.TrainingData,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(ImageData.Training.TrainingData(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
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
N=length(ImageData.Training.TrainingData(1,:)); % number of images
Xmean=mean(ImageData.Training.TrainingData,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(ImageData.Training.TrainingData(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
tic
S_=(1/N)*(Xprime'*Xprime); % data covariance matrix
[eigVec_,eigVal_]=eig(S_); % Finding lambda_m and u_m as in Q1
U416=Xprime*eigVec_; %matrix of 416 largest eigenvectors
U416=normc(U416); % normalize U416
elapsedtime=toc
Eigenvalues=diag(eigVal_);

%% 5.
clear all
close all
clc
load face.mat
% The image data is organized by training and testing data and also by person

ImageData.Training=struct('Person1',{X(:,1:8)},'Person2',{X(:,11:18)},'Person3',{X(:,21:28)},'Person4',{X(:,31:38)},'Person5',{X(:,41:48)},'Person6',{X(:,51:58)},'Person7',{X(:,61:68)},'Person8',{X(:,71:78)},'Person9',{X(:,81:88)},'Person10',{X(:,91:98)}, ...
    'Person11',{X(:,101:108)},'Person12',{X(:,111:118)},'Person13',{X(:,121:128)},'Person14',{X(:,131:138)},'Person15',{X(:,141:148)},'Person16',{X(:,151:158)},'Person17',{X(:,161:168)},'Person18',{X(:,171:178)},'Person19',{X(:,181:188)},'Person20',{X(:,191:198)}, ...
    'Person21',{X(:,201:208)},'Person22',{X(:,211:218)},'Person23',{X(:,221:228)},'Person24',{X(:,231:238)},'Person25',{X(:,241:248)},'Person26',{X(:,251:258)},'Person27',{X(:,261:268)},'Person28',{X(:,271:278)},'Person29',{X(:,281:288)},'Person30',{X(:,291:298)}, ...
    'Person31',{X(:,301:308)},'Person32',{X(:,311:318)},'Person33',{X(:,321:328)},'Person34',{X(:,331:338)},'Person35',{X(:,341:348)},'Person36',{X(:,351:358)},'Person37',{X(:,361:368)},'Person38',{X(:,371:378)},'Person39',{X(:,381:388)},'Person40',{X(:,391:398)}, ...
    'Person41',{X(:,401:408)},'Person42',{X(:,411:418)},'Person43',{X(:,421:428)},'Person44',{X(:,431:438)},'Person45',{X(:,441:448)},'Person46',{X(:,451:458)},'Person47',{X(:,461:468)},'Person48',{X(:,471:478)},'Person49',{X(:,481:488)},'Person50',{X(:,491:498)}, ...
    'Person51',{X(:,501:508)},'Person52',{X(:,511:518)},'TrainingData',{cat(2,X(:,1:8),X(:,11:18),X(:,21:28),X(:,31:38),X(:,41:48),X(:,51:58),X(:,61:68),X(:,71:78),X(:,81:88),X(:,91:98),X(:,101:108),X(:,111:118),X(:,121:128),X(:,131:138),X(:,141:148),X(:,151:158), ...
    X(:,161:168),X(:,171:178),X(:,181:188),X(:,191:198),X(:,201:208),X(:,211:218),X(:,221:228),X(:,231:238),X(:,241:248),X(:,251:258),X(:,261:268),X(:,271:278),X(:,281:288),X(:,291:298),X(:,301:308),X(:,311:318),X(:,321:328),X(:,331:338),X(:,341:348),X(:,351:358), ...
    X(:,361:368),X(:,371:378),X(:,381:388),X(:,391:398),X(:,401:408),X(:,411:418),X(:,421:428),X(:,431:438),X(:,441:448),X(:,451:458),X(:,461:468),X(:,471:478),X(:,481:488),X(:,491:498),X(:,501:508),X(:,511:518))});

ImageData.Testing=struct('Person1',{X(:,9:10)},'Person2',{X(:,19:20)},'Person3',{X(:,29:30)},'Person4',{X(:,39:40)},'Person5',{X(:,49:50)},'Person6',{X(:,59:60)},'Person7',{X(:,69:70)},'Person8',{X(:,79:80)},'Person9',{X(:,89:90)},'Person10',{X(:,99:100)}, ...
    'Person11',{X(:,109:110)},'Person12',{X(:,119:120)},'Person13',{X(:,129:130)},'Person14',{X(:,139:140)},'Person15',{X(:,149:150)},'Person16',{X(:,159:160)},'Person17',{X(:,169:170)},'Person18',{X(:,179:180)},'Person19',{X(:,189:190)},'Person20',{X(:,199:200)}, ...
    'Person21',{X(:,209:210)},'Person22',{X(:,219:220)},'Person23',{X(:,229:230)},'Person24',{X(:,239:240)},'Person25',{X(:,249:250)},'Person26',{X(:,259:260)},'Person27',{X(:,269:270)},'Person28',{X(:,279:280)},'Person29',{X(:,289:290)},'Person30',{X(:,299:300)}, ...
    'Person31',{X(:,309:310)},'Person32',{X(:,319:320)},'Person33',{X(:,329:330)},'Person34',{X(:,339:340)},'Person35',{X(:,349:350)},'Person36',{X(:,359:360)},'Person37',{X(:,369:370)},'Person38',{X(:,379:380)},'Person39',{X(:,389:390)},'Person40',{X(:,399:400)}, ...
    'Person41',{X(:,409:410)},'Person42',{X(:,419:420)},'Person43',{X(:,429:430)},'Person44',{X(:,439:440)},'Person45',{X(:,449:450)},'Person46',{X(:,459:460)},'Person47',{X(:,469:470)},'Person48',{X(:,479:480)},'Person49',{X(:,489:490)},'Person50',{X(:,499:500)}, ...
    'Person51',{X(:,509:510)},'Person52',{X(:,519:520)},'TestingData',{cat(2,X(:,9:10),X(:,19:20),X(:,29:30),X(:,39:40),X(:,49:50),X(:,59:60),X(:,69:70),X(:,79:80),X(:,89:90),X(:,99:100),X(:,109:110),X(:,119:120),X(:,129:130),X(:,139:140),X(:,149:150),X(:,159:160), ...
    X(:,169:170),X(:,179:180),X(:,189:190),X(:,199:200),X(:,209:210),X(:,219:220),X(:,229:230),X(:,239:240),X(:,249:250),X(:,259:260),X(:,269:270),X(:,279:280),X(:,289:290),X(:,299:300),X(:,309:310),X(:,319:320),X(:,329:330),X(:,339:340),X(:,349:350),X(:,359:360), ...
    X(:,369:370),X(:,379:380),X(:,389:390),X(:,399:400),X(:,409:410),X(:,419:420),X(:,429:430),X(:,439:440),X(:,449:450),X(:,459:460),X(:,469:470),X(:,479:480),X(:,489:490),X(:,499:500),X(:,509:510),X(:,519:520))});
clearvars X l % saving memory
N=length(ImageData.Training.TrainingData(1,:)); % number of images
Xmean=mean(ImageData.Training.TrainingData,2); % the mean image of the training images
Xmean=repmat(Xmean,[1 N]);
Xprime=(ImageData.Training.TrainingData(:,1:N)-Xmean); % Xprime=[...,Xi - Xmean,...]
S_=(1/N)*(Xprime'*Xprime); % data covariance matrix
[eigVec_,eigVal_]=eig(S_); % Finding lambda_m and u_m as in Q1
U416=Xprime*eigVec_; %matrix of 416 largest eigenvectors
U416=normc(U416); % normalize U416
%% Test image 1: Person 1, 1st training Image
D=length(ImageData.Training.Person1(:,1));
X1=ImageData.Training.Person1(:,1);
X1=reshape(X1,56,46);
figure
subplot(2,3,1)
imagesc(X1)
set(gca,'FontSize',20)
title('Original Image')
colormap(gray)
%%%% 1 base
X1=ImageData.Training.Person1(:,1);
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
title('1 PCA base')
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
title('5 PCA bases')
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
title('10 PCA bases')
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
title('20 PCA bases')
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
title('30 PCA bases')
colormap(gray)
%% Test image 2: Person 26, 5th training Image 
D=length(ImageData.Training.Person26(:,5));
X26=ImageData.Training.Person26(:,5);
X26=reshape(X26,56,46);
figure
subplot(2,3,1)
imagesc(X26)
set(gca,'FontSize',20)
title('Original Image')
colormap(gray)
%%%% 1 base
X26=ImageData.Training.Person26(:,5);
X26mean=mean(X26);
X26mean=repmat(X26mean,[D 1]);
X26prime=(X26-X26mean);
S26=X26prime*X26prime';
[eigVec26,eigVal26]=eig(S26);
[~,sortIndex] = sort(diag(eigVal26),'descend');
maxIndex = sortIndex(1:1);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec26(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA26=U416*eigVec';
PCA26mean=mean(PCA26(:,maxIndex),2);
PCA26mean=reshape(PCA26mean,56,46);
subplot(2,3,2)
imagesc(PCA26mean)
set(gca,'FontSize',20)
title('1 PCA base')
colormap(gray)
%%% 5 bases
X26mean=mean(X26);
X26mean=repmat(X26mean,[D 5]);
X26=repmat(X26,[1 5]);
X26prime=(X26-X26mean);
S26=X26prime*X26prime';
[eigVec26,eigVal26]=eig(S26);
[~,sortIndex] = sort(diag(eigVal26),'descend');
maxIndex = sortIndex(1:5);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec26(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA26=U416*eigVec';
PCA26mean=mean(PCA26(:,maxIndex),2);
PCA26mean=reshape(PCA26mean,56,46);
subplot(2,3,3)
imagesc(PCA26mean)
set(gca,'FontSize',20)
title('5 PCA bases')
colormap(gray)
%%%%% 10 bases
X26mean=mean(X26);
X26mean=repmat(X26mean,[D 10]);
X26=repmat(X26,[1 10]);
X26prime=(X26-X26mean);
S26=X26prime*X26prime';
[eigVec26,eigVal26]=eig(S26);
[~,sortIndex] = sort(diag(eigVal26),'descend');
maxIndex = sortIndex(1:10);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec26(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA26=U416*eigVec';
PCA26mean=mean(PCA26(:,maxIndex),2);
PCA26mean=reshape(PCA26mean,56,46);
subplot(2,3,4)
imagesc(PCA26mean)
set(gca,'FontSize',20)
title('10 PCA bases')
colormap(gray)
%%%%% 20 bases
X26mean=mean(X26);
X26mean=repmat(X26mean,[D 20]);
X26=repmat(X26,[1 20]);
X26prime=(X26-X26mean);
S26=X26prime*X26prime';
[eigVec26,eigVal26]=eig(S26);
[~,sortIndex] = sort(diag(eigVal26),'descend');
maxIndex = sortIndex(1:20);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec26(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA26=U416*eigVec';
PCA26mean=mean(PCA26(:,maxIndex),2);
PCA26mean=reshape(PCA26mean,56,46);
subplot(2,3,5)
imagesc(PCA26mean)
set(gca,'FontSize',20)
title('20 PCA bases')
colormap(gray)
%%%%% 30 bases
X26mean=mean(X26);
X26mean=repmat(X26mean,[D 30]);
X26=repmat(X26,[1 30]);
X26prime=(X26-X26mean);
S26=X26prime*X26prime';
[eigVec26,eigVal26]=eig(S26);
[~,sortIndex] = sort(diag(eigVal26),'descend');
maxIndex = sortIndex(1:30);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec26(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA26=U416*eigVec';
PCA26mean=mean(PCA26(:,maxIndex),2);
PCA26mean=reshape(PCA26mean,56,46);
subplot(2,3,6)
imagesc(PCA26mean)
set(gca,'FontSize',20)
title('30 PCA bases')
colormap(gray)
%% Test image 3: Person 52, 2nd testing Image
D=length(ImageData.Testing.Person52(:,2));
X52=ImageData.Testing.Person52(:,2);
X52=reshape(X52,56,46);
figure
subplot(2,3,1)
imagesc(X52)
set(gca,'FontSize',20)
title('Original Image')
colormap(gray)
%%%% 1 base
X52=ImageData.Testing.Person52(:,2);
X52mean=mean(X52);
X52mean=repmat(X52mean,[D 1]);
X52prime=(X52-X52mean);
S52=X52prime*X52prime';
[eigVec52,eigVal52]=eig(S52);
[~,sortIndex] = sort(diag(eigVal52),'descend');
maxIndex = sortIndex(1:1);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec52(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA52=U416*eigVec';
PCA52mean=mean(PCA52(:,maxIndex),2);
PCA52mean=reshape(PCA52mean,56,46);
subplot(2,3,2)
imagesc(PCA52mean)
set(gca,'FontSize',20)
title('1 PCA base')
colormap(gray)
%%% 5 bases
X52mean=mean(X52);
X52mean=repmat(X52mean,[D 5]);
X52=repmat(X52,[1 5]);
X52prime=(X52-X52mean);
S52=X52prime*X52prime';
[eigVec52,eigVal52]=eig(S52);
[~,sortIndex] = sort(diag(eigVal52),'descend');
maxIndex = sortIndex(1:5);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec52(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA52=U416*eigVec';
PCA52mean=mean(PCA52(:,maxIndex),2);
PCA52mean=reshape(PCA52mean,56,46);
subplot(2,3,3)
imagesc(PCA52mean)
set(gca,'FontSize',20)
title('5 PCA bases')
colormap(gray)
%%%%% 10 bases
X52mean=mean(X52);
X52mean=repmat(X52mean,[D 10]);
X52=repmat(X52,[1 10]);
X52prime=(X52-X52mean);
S52=X52prime*X52prime';
[eigVec52,eigVal52]=eig(S52);
[~,sortIndex] = sort(diag(eigVal52),'descend');
maxIndex = sortIndex(1:10);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec52(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA52=U416*eigVec';
PCA52mean=mean(PCA52(:,maxIndex),2);
PCA52mean=reshape(PCA52mean,56,46);
subplot(2,3,4)
imagesc(PCA52mean)
set(gca,'FontSize',20)
title('10 PCA bases')
colormap(gray)
%%%%% 20 bases
X52mean=mean(X52);
X52mean=repmat(X52mean,[D 20]);
X52=repmat(X52,[1 20]);
X52prime=(X52-X52mean);
S52=X52prime*X52prime';
[eigVec52,eigVal52]=eig(S52);
[~,sortIndex] = sort(diag(eigVal52),'descend');
maxIndex = sortIndex(1:20);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec52(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA52=U416*eigVec';
PCA52mean=mean(PCA52(:,maxIndex),2);
PCA52mean=reshape(PCA52mean,56,46);
subplot(2,3,5)
imagesc(PCA52mean)
set(gca,'FontSize',20)
title('20 PCA bases')
colormap(gray)
%%%%% 30 bases
X52mean=mean(X52);
X52mean=repmat(X52mean,[D 30]);
X52=repmat(X52,[1 30]);
X52prime=(X52-X52mean);
S52=X52prime*X52prime';
[eigVec52,eigVal52]=eig(S52);
[~,sortIndex] = sort(diag(eigVal52),'descend');
maxIndex = sortIndex(1:30);
eigVec=zeros(D,D);
eigVec(:,maxIndex)=eigVec52(:,maxIndex);
eigVec=eigVec(:,end-415:end);
PCA52=U416*eigVec';
PCA52mean=mean(PCA52(:,maxIndex),2);
PCA52mean=reshape(PCA52mean,56,46);
subplot(2,3,6)
imagesc(PCA52mean)
set(gca,'FontSize',20)
title('30 PCA bases')
colormap(gray)

%% 6., Method 1
clear all
clearvars
close all
load face.mat
% Image data partitioning and organizing into field arrays
% The image data is organized by training and testing data and thereafter by person

ImageData.Training=struct('Person1',{X(:,1:8)},'Person2',{X(:,11:18)},'Person3',{X(:,21:28)},'Person4',{X(:,31:38)},'Person5',{X(:,41:48)},'Person6',{X(:,51:58)},'Person7',{X(:,61:68)},'Person8',{X(:,71:78)},'Person9',{X(:,81:88)},'Person10',{X(:,91:98)}, ...
    'Person11',{X(:,101:108)},'Person12',{X(:,111:118)},'Person13',{X(:,121:128)},'Person14',{X(:,131:138)},'Person15',{X(:,141:148)},'Person16',{X(:,151:158)},'Person17',{X(:,161:168)},'Person18',{X(:,171:178)},'Person19',{X(:,181:188)},'Person20',{X(:,191:198)}, ...
    'Person21',{X(:,201:208)},'Person22',{X(:,211:218)},'Person23',{X(:,221:228)},'Person24',{X(:,231:238)},'Person25',{X(:,241:248)},'Person26',{X(:,251:258)},'Person27',{X(:,261:268)},'Person28',{X(:,271:278)},'Person29',{X(:,281:288)},'Person30',{X(:,291:298)}, ...
    'Person31',{X(:,301:308)},'Person32',{X(:,311:318)},'Person33',{X(:,321:328)},'Person34',{X(:,331:338)},'Person35',{X(:,341:348)},'Person36',{X(:,351:358)},'Person37',{X(:,361:368)},'Person38',{X(:,371:378)},'Person39',{X(:,381:388)},'Person40',{X(:,391:398)}, ...
    'Person41',{X(:,401:408)},'Person42',{X(:,411:418)},'Person43',{X(:,421:428)},'Person44',{X(:,431:438)},'Person45',{X(:,441:448)},'Person46',{X(:,451:458)},'Person47',{X(:,461:468)},'Person48',{X(:,471:478)},'Person49',{X(:,481:488)},'Person50',{X(:,491:498)}, ...
    'Person51',{X(:,501:508)},'Person52',{X(:,511:518)},'TrainingData',{cat(2,X(:,1:8),X(:,11:18),X(:,21:28),X(:,31:38),X(:,41:48),X(:,51:58),X(:,61:68),X(:,71:78),X(:,81:88),X(:,91:98),X(:,101:108),X(:,111:118),X(:,121:128),X(:,131:138),X(:,141:148),X(:,151:158), ...
    X(:,161:168),X(:,171:178),X(:,181:188),X(:,191:198),X(:,201:208),X(:,211:218),X(:,221:228),X(:,231:238),X(:,241:248),X(:,251:258),X(:,261:268),X(:,271:278),X(:,281:288),X(:,291:298),X(:,301:308),X(:,311:318),X(:,321:328),X(:,331:338),X(:,341:348),X(:,351:358), ...
    X(:,361:368),X(:,371:378),X(:,381:388),X(:,391:398),X(:,401:408),X(:,411:418),X(:,421:428),X(:,431:438),X(:,441:448),X(:,451:458),X(:,461:468),X(:,471:478),X(:,481:488),X(:,491:498),X(:,501:508),X(:,511:518))});

ImageData.Testing=struct('Person1',{X(:,9:10)},'Person2',{X(:,19:20)},'Person3',{X(:,29:30)},'Person4',{X(:,39:40)},'Person5',{X(:,49:50)},'Person6',{X(:,59:60)},'Person7',{X(:,69:70)},'Person8',{X(:,79:80)},'Person9',{X(:,89:90)},'Person10',{X(:,99:100)}, ...
    'Person11',{X(:,109:110)},'Person12',{X(:,119:120)},'Person13',{X(:,129:130)},'Person14',{X(:,139:140)},'Person15',{X(:,149:150)},'Person16',{X(:,159:160)},'Person17',{X(:,169:170)},'Person18',{X(:,179:180)},'Person19',{X(:,189:190)},'Person20',{X(:,199:200)}, ...
    'Person21',{X(:,209:210)},'Person22',{X(:,219:220)},'Person23',{X(:,229:230)},'Person24',{X(:,239:240)},'Person25',{X(:,249:250)},'Person26',{X(:,259:260)},'Person27',{X(:,269:270)},'Person28',{X(:,279:280)},'Person29',{X(:,289:290)},'Person30',{X(:,299:300)}, ...
    'Person31',{X(:,309:310)},'Person32',{X(:,319:320)},'Person33',{X(:,329:330)},'Person34',{X(:,339:340)},'Person35',{X(:,349:350)},'Person36',{X(:,359:360)},'Person37',{X(:,369:370)},'Person38',{X(:,379:380)},'Person39',{X(:,389:390)},'Person40',{X(:,399:400)}, ...
    'Person41',{X(:,409:410)},'Person42',{X(:,419:420)},'Person43',{X(:,429:430)},'Person44',{X(:,439:440)},'Person45',{X(:,449:450)},'Person46',{X(:,459:460)},'Person47',{X(:,469:470)},'Person48',{X(:,479:480)},'Person49',{X(:,489:490)},'Person50',{X(:,499:500)}, ...
    'Person51',{X(:,509:510)},'Person52',{X(:,519:520)},'TestingData',{cat(2,X(:,9:10),X(:,19:20),X(:,29:30),X(:,39:40),X(:,49:50),X(:,59:60),X(:,69:70),X(:,79:80),X(:,89:90),X(:,99:100),X(:,109:110),X(:,119:120),X(:,129:130),X(:,139:140),X(:,149:150),X(:,159:160), ...
    X(:,169:170),X(:,179:180),X(:,189:190),X(:,199:200),X(:,209:210),X(:,219:220),X(:,229:230),X(:,239:240),X(:,249:250),X(:,259:260),X(:,269:270),X(:,279:280),X(:,289:290),X(:,299:300),X(:,309:310),X(:,319:320),X(:,329:330),X(:,339:340),X(:,349:350),X(:,359:360), ...
    X(:,369:370),X(:,379:380),X(:,389:390),X(:,399:400),X(:,409:410),X(:,419:420),X(:,429:430),X(:,439:440),X(:,449:450),X(:,459:460),X(:,469:470),X(:,479:480),X(:,489:490),X(:,499:500),X(:,509:510),X(:,519:520))});
clearvars X l
D=length(ImageData.Training.Person1);
XCprime=zeros(D,8,52);
N=length(ImageData.Training.Person1(1,:));

for c=1:52
    XCn(:,:,c)=ImageData.Training.TrainingData(:,(c-1)*length(XCprime(1,:,1))+1:c*length(XCprime(1,:,1)));
    XCmean(:,c)=mean(XCn(:,:,c),2); 
    XCmean_tmp=repmat(XCmean(:,c),[1 length(XCprime(1,:,1))]);
    XCprime(:,:,c)=XCn(:,:,c)-XCmean_tmp;
    S8C(:,:,c)=(1/length(XCprime(1,:,1)))*XCprime(:,:,c)'*XCprime(:,:,c);
    [eigVec8C(:,:,c),eigVal8C(:,:,c)]=eig(S8C(:,:,c));
    [sortedValues(:,c),sortIndex(:,c)] = sort(diag(eigVal8C(:,:,c)),'descend');
    U8(:,:,c)=XCprime(:,:,c)*eigVec8C(:,:,c);
    U8(:,:,c)=normc(U8(:,:,c));
end

% Method 1: Test image projection onto eigen-subspace of each class
tic
for img=1:104
    for c=1:52
        Ztest=U8(:,:,c)'*(ImageData.Testing.TestingData(:,img)-XCmean(:,c));
        Xtilda=XCmean(:,c)+U8(:,:,c)*Ztest;
        Xdist(img,c)=norm(ImageData.Testing.TestingData(:,img)-Xtilda);
    end
    [~,Predict1(img)]=min(Xdist(img,:)); %
end
Elapsedtime=toc
for img=1:2:104
    label(img:img+1)=ceil(img/2);
end

Predict1_Recognition_acc=mean(label==Predict1); % Classification accuracy 
idx = sub2ind([52, 52], label, Predict1) ;
conf = zeros(52) ;
conf = vl_binsum(conf, ones(size(idx)), idx) ;

imagesc(conf) ;
set(gca,'FontSize',20)
title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * Predict1_Recognition_acc) );
%% 7., Method 2
clear all
clearvars
close all
load face.mat
% Image data partitioning and organizing into field arrays
% The image data is organized by training and testing data and thereafter by person

ImageData.Training=struct('Person1',{X(:,1:8)},'Person2',{X(:,11:18)},'Person3',{X(:,21:28)},'Person4',{X(:,31:38)},'Person5',{X(:,41:48)},'Person6',{X(:,51:58)},'Person7',{X(:,61:68)},'Person8',{X(:,71:78)},'Person9',{X(:,81:88)},'Person10',{X(:,91:98)}, ...
    'Person11',{X(:,101:108)},'Person12',{X(:,111:118)},'Person13',{X(:,121:128)},'Person14',{X(:,131:138)},'Person15',{X(:,141:148)},'Person16',{X(:,151:158)},'Person17',{X(:,161:168)},'Person18',{X(:,171:178)},'Person19',{X(:,181:188)},'Person20',{X(:,191:198)}, ...
    'Person21',{X(:,201:208)},'Person22',{X(:,211:218)},'Person23',{X(:,221:228)},'Person24',{X(:,231:238)},'Person25',{X(:,241:248)},'Person26',{X(:,251:258)},'Person27',{X(:,261:268)},'Person28',{X(:,271:278)},'Person29',{X(:,281:288)},'Person30',{X(:,291:298)}, ...
    'Person31',{X(:,301:308)},'Person32',{X(:,311:318)},'Person33',{X(:,321:328)},'Person34',{X(:,331:338)},'Person35',{X(:,341:348)},'Person36',{X(:,351:358)},'Person37',{X(:,361:368)},'Person38',{X(:,371:378)},'Person39',{X(:,381:388)},'Person40',{X(:,391:398)}, ...
    'Person41',{X(:,401:408)},'Person42',{X(:,411:418)},'Person43',{X(:,421:428)},'Person44',{X(:,431:438)},'Person45',{X(:,441:448)},'Person46',{X(:,451:458)},'Person47',{X(:,461:468)},'Person48',{X(:,471:478)},'Person49',{X(:,481:488)},'Person50',{X(:,491:498)}, ...
    'Person51',{X(:,501:508)},'Person52',{X(:,511:518)},'TrainingData',{cat(2,X(:,1:8),X(:,11:18),X(:,21:28),X(:,31:38),X(:,41:48),X(:,51:58),X(:,61:68),X(:,71:78),X(:,81:88),X(:,91:98),X(:,101:108),X(:,111:118),X(:,121:128),X(:,131:138),X(:,141:148),X(:,151:158), ...
    X(:,161:168),X(:,171:178),X(:,181:188),X(:,191:198),X(:,201:208),X(:,211:218),X(:,221:228),X(:,231:238),X(:,241:248),X(:,251:258),X(:,261:268),X(:,271:278),X(:,281:288),X(:,291:298),X(:,301:308),X(:,311:318),X(:,321:328),X(:,331:338),X(:,341:348),X(:,351:358), ...
    X(:,361:368),X(:,371:378),X(:,381:388),X(:,391:398),X(:,401:408),X(:,411:418),X(:,421:428),X(:,431:438),X(:,441:448),X(:,451:458),X(:,461:468),X(:,471:478),X(:,481:488),X(:,491:498),X(:,501:508),X(:,511:518))});

ImageData.Testing=struct('Person1',{X(:,9:10)},'Person2',{X(:,19:20)},'Person3',{X(:,29:30)},'Person4',{X(:,39:40)},'Person5',{X(:,49:50)},'Person6',{X(:,59:60)},'Person7',{X(:,69:70)},'Person8',{X(:,79:80)},'Person9',{X(:,89:90)},'Person10',{X(:,99:100)}, ...
    'Person11',{X(:,109:110)},'Person12',{X(:,119:120)},'Person13',{X(:,129:130)},'Person14',{X(:,139:140)},'Person15',{X(:,149:150)},'Person16',{X(:,159:160)},'Person17',{X(:,169:170)},'Person18',{X(:,179:180)},'Person19',{X(:,189:190)},'Person20',{X(:,199:200)}, ...
    'Person21',{X(:,209:210)},'Person22',{X(:,219:220)},'Person23',{X(:,229:230)},'Person24',{X(:,239:240)},'Person25',{X(:,249:250)},'Person26',{X(:,259:260)},'Person27',{X(:,269:270)},'Person28',{X(:,279:280)},'Person29',{X(:,289:290)},'Person30',{X(:,299:300)}, ...
    'Person31',{X(:,309:310)},'Person32',{X(:,319:320)},'Person33',{X(:,329:330)},'Person34',{X(:,339:340)},'Person35',{X(:,349:350)},'Person36',{X(:,359:360)},'Person37',{X(:,369:370)},'Person38',{X(:,379:380)},'Person39',{X(:,389:390)},'Person40',{X(:,399:400)}, ...
    'Person41',{X(:,409:410)},'Person42',{X(:,419:420)},'Person43',{X(:,429:430)},'Person44',{X(:,439:440)},'Person45',{X(:,449:450)},'Person46',{X(:,459:460)},'Person47',{X(:,469:470)},'Person48',{X(:,479:480)},'Person49',{X(:,489:490)},'Person50',{X(:,499:500)}, ...
    'Person51',{X(:,509:510)},'Person52',{X(:,519:520)},'TestingData',{cat(2,X(:,9:10),X(:,19:20),X(:,29:30),X(:,39:40),X(:,49:50),X(:,59:60),X(:,69:70),X(:,79:80),X(:,89:90),X(:,99:100),X(:,109:110),X(:,119:120),X(:,129:130),X(:,139:140),X(:,149:150),X(:,159:160), ...
    X(:,169:170),X(:,179:180),X(:,189:190),X(:,199:200),X(:,209:210),X(:,219:220),X(:,229:230),X(:,239:240),X(:,249:250),X(:,259:260),X(:,269:270),X(:,279:280),X(:,289:290),X(:,299:300),X(:,309:310),X(:,319:320),X(:,329:330),X(:,339:340),X(:,349:350),X(:,359:360), ...
    X(:,369:370),X(:,379:380),X(:,389:390),X(:,399:400),X(:,409:410),X(:,419:420),X(:,429:430),X(:,439:440),X(:,449:450),X(:,459:460),X(:,469:470),X(:,479:480),X(:,489:490),X(:,499:500),X(:,509:510),X(:,519:520))});
clearvars X l

% Method 2, create principal eigen-subspace over all data
Xn=ImageData.Training.TrainingData;
Xmean=mean(Xn,2);
Xmean_tmp=repmat(Xmean,[1 416]);
Xprime=Xn-Xmean_tmp;

S416=(1/416)*(Xprime')*Xprime;
[eigVec8,eigVal8]=eig(S416);
U416=Xprime*eigVec8;
U416=normc(U416);
%% Method 2, projections of class means comparison,||Ztest-ZCmean||
% Method 2: calculate the projection of the cth class data mean
tic
for c=1:52
    XCn(:,:,c)=ImageData.Training.TrainingData(:,(c-1)*8+1:c*8);
    XCmean(:,c)=mean(XCn(:,:,c),2); 
    ZC(:,:,c)=U416'*((XCn(:,:,c))-repmat(Xmean,[1 8]));
    temp = ZC(:,:,c);
    ZCmean(:,c)=mean(ZC(:,:,c),2);
end

% Method 2: Test image projection onto eigen-subspace, Z, and calculate difference vector between projections of test point and mean, ||Ztest-ZCmean||
for img=1:104
    for c=1:52
        Ztest=U416'*(ImageData.Testing.TestingData(:,img)-Xmean);
        Zdist(img,c)=norm(Ztest-ZCmean(:,c));
        [~,Predict2(img)]=min(Zdist(img,:));
    end
end
Elapsedtime=toc
for img=1:2:104
    label(img:img+1)=ceil(img/2);
end

Predict2_Recognition_acc=mean(label==Predict2); % Classification accuracy 
idx = sub2ind([52, 52], label, Predict2) ;
conf = zeros(52) ;
conf = vl_binsum(conf, ones(size(idx)), idx) ;

imagesc(conf) ;
set(gca,'FontSize',20)
title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * Predict2_Recognition_acc) ) ;
%% Method 2, NN comparison
% Method 2: Test image projection onto eigen-subspace and NN comparison
tic
for imgtest=1:104
    for imgtrain=1:416
        Ztest=U416'*(ImageData.Testing.TestingData(:,imgtest)-Xmean);
        Ztrain=U416'*(ImageData.Training.TrainingData(:,imgtrain)-Xmean);
        Zdist(imgtest,imgtrain)=norm(Ztest-Ztrain);
        [~,Predict2NN(imgtest)]=min(Zdist(imgtest,:));
        Predict2NN(imgtest)=ceil(Predict2NN(imgtest)/8);
    end
end
Elapsedtime=toc
for img=1:2:104
    label(img:img+1)=ceil(img/2);
end

Predict2NN_Recognition_acc=mean(label==Predict2NN);
idx = sub2ind([52, 52], label, Predict2NN) ;
conf = zeros(52) ;
conf = vl_binsum(conf, ones(size(idx)), idx) ;

imagesc(conf);
set(gca,'FontSize',20)
title(sprintf('Confusion matrix (%.2f %% accuracy)', 100 * Predict2NN_Recognition_acc) ) ;
