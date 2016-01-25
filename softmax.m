function s=softmax(x,dim)

    x=bsxfun(@minus,x,max(x,[],dim));
    s=exp(x);
    s=bsxfun(@rdivide,s,sum(s,dim));

end
