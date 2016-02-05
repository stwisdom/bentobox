function ce_or_grad=bb_crossEntropy(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    Y=args.Y;
    Yhat=args.Yhat;
    dim_time=args.dimt;
    dim_marg=args.dimm;
    
    T=size(Yhat,dim_time);
    
    if size(Y,dim_time)~=T
        error('Yhat and Y need to have same number of elements!');
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            logYhat=log(realmin('single')+Yhat);
%             % get rid of -Infs if they exist:
%             logYhat(logYhat<log(realmin('single')))=log(realmin('single'));
            ce=-sum(fxns.pagefun_marg(Y,logYhat,dim_marg),dim_time)./T;
            
            ce_or_grad=ce;
            
        case 2
            %%% backward pass
            
            ce=args.output;
            
            switch(parentLabelGrad)
  
                case 'Yhat'
                    Yhat(Yhat==0)=realmin('single');
                    grad=-bsxfun(@rdivide,Y,Yhat)./T;

            end
            
            ce_or_grad=grad;
            
    end
    