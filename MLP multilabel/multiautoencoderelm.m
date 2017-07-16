function data_final=multiautoencoderelm(data)
%this function is an autoencoder with 2 layer using ELM techniques
%data=csvread('scene-trainnew.csv');
len=size(data,1);
%  ix = randperm(len);
%  data = data(ix,:); % shuffle the data
 %data
 ln=6;  %no of unique label
 nof=size(data,2)-6;%no. of features
 L=250;%no. of nodes
 col=size(data,2);
features=data(:,1:nof);
labels=data(:,nof+1:col);
% labels=zeros(len,ln);
% for x=1:len
%     labels(x,label(x))=1;
% end

w=rand(L,nof);  %random weight matrix
b=rand(L,1);
h=[];
for x=1:len
    Y=w*features(x,:)'+b;
    Y=sigmf(Y,[1 0]);
    h(:,x)=Y;
end
h=h';
h;

beta=pinv(h)*features;

final_features=features*pinv(beta);
size(final_features);
%final_features(:,L+1)=label;
%final_features;






nof2=L;%no. of features
 L2=150;%no. of nodes
features2=final_features;
size(features2,2);

w=rand(L2,nof2);  %random weight matrix
b2=rand(L2,1);
h2=[];
for x=1:len
    Y=w*features2(x,:)'+b2;
    Y=sigmf(Y,[1 0]);
    h2(:,x)=Y;
end
h2=h2';
h2;

beta2=pinv(h2)*features2;

final_features2=features2*pinv(beta2);
size(final_features2);
% final_features2(:,L2+1)=label;
% final_features2;
size(final_features2)
size(labels)
data_final=[final_features2,labels];
 data_final;
 size(data_final)


end

