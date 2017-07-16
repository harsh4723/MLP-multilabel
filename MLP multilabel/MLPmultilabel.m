%muktilabel prediction using MLP
data=csvread('yeast-trainnew.csv');%change data to dat1 for autoencoder
%data=multiautoencoderelm(dat1);%uncomment to use autoencoder
size(data,2);

len=size(data,1);
% %data
%  %ix = randperm(len);
%  %data = data(ix,:); %t shuffle the data
%  %data
  ln=6;  %no of unique label
 nof=size(data,2)-6 ;%no. of features
feature_train=data(:,1:nof);
%size(features);
nofcol=size(data,2);
label_train=data(:,nof+1:nofcol);

 ntrain=len;

 net = fitnet([10,15]);
 net = train(net,feature_train',label_train');
 %view(net)
 label_got=net(feature_train');
 label_got=label_got'
 
 
 
 datatest=csvread('yeast-testnew.csv');%change datatest to datatest1 for autoencoder
 %datatest=multiautoencoderelm(datatest1);%uncomment to use autoencoder
 ntest=size(datatest,1);
 
 feature_test=datatest(:,1:nof);
 nofcol2=size(datatest,2);
 label_test=datatest(:,nof+1:nofcol2);
 
 y = net(feature_test');
 
final=y';

final1=MLPupdate(label_got,label_train,final);
 final1=round(final1');

c=0;
for i=1:ntest
    s=0;
    for j=1:ln
        s=s+xor(final1(i,j),label_test(i,j));
    end
    c=c+(s/ln);
end
hloss=c/ntest