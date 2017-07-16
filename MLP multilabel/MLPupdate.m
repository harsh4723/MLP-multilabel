function predicted=MLPupdate(feature_train,label_train,feature_test)
%this function is multilabel perceptron for threshholding
feature_train';

 net = fitnet([10,15]);
 net = train(net,feature_train',label_train');
 %view(net)
 
 y = net(feature_test');
%predicted=round(y)
predicted=y;
% c=0;
% for x=1:ntest
%     if(predicted(x)==label_test(x))
%         c=c+1;
%     end
% end
% accu=c/ntest
end
 %perf = perform(net,y,label_test')