function xsum_or_grad=bb_sumDim(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x=args.x;
    dim=args.dim;
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            xsum=sum(x,dim);
            
            xsum_or_grad=xsum;
            
        case 2
            %%% backward pass
            
            D_wrt_xsum=args.D_wrt_thisNode;
            
            grad=bsxfun(@times,D_wrt_xsum,fxns.gpuArray(ones(size(x))));
            
            xsum_or_grad=grad;
            
    end
    