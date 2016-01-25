function x1Minusx2_or_grad=bb_minus(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x1=args.x1;
    x2=args.x2;
    
    x1sz=size(x1);
    x2sz=size(x2);
    
    %check that x1 and x2 are compatible sizes:
    x1ndims=length(x1sz);
    x2ndims=length(x2sz);
    if x1ndims<x2ndims
        x1sz=[x1sz,ones(1,x2ndims-x1ndims)];
    else
        x2sz=[x2sz,ones(1,x1ndims-x2ndims)];
    end
    x1sz_neq_x2sz=(x1sz~=x2sz);
    if sum(~(x1sz(x1sz_neq_x2sz)==1 | x2sz(x1sz_neq_x2sz)==1))
        error('x1 and x2 need to be compatible sizes!');
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            x1Minusx2=bsxfun(@minus,x1,x2);
            
            x1Minusx2_or_grad=x1Minusx2;
            
        case 2
            %%% backward pass
            
            D_wrt_thisNode=args.D_wrt_thisNode;
            Dsz=size(D_wrt_thisNode);
            Dndims=length(Dsz);
            
            switch(parentLabelGrad)
  
                case 'x1'
                    if x1ndims<Dndims
                        x1sz=[x1sz,ones(1,Dndims-x1ndims)];
                    end
                    
                    ieq1=find(x1sz==1);
                    grad=bsxfun(@times,D_wrt_thisNode,ones(size(x1)));
                    for ii=ieq1
                        grad=sum(grad,ii);
                    end
                    
                case 'x2'
                    if x2ndims<Dndims
                        x2sz=[x2sz,ones(1,Dndims-x2ndims)];
                    end
                    
                    ieq1=find(x2sz==1);
                    grad=bsxfun(@times,D_wrt_thisNode,-ones(size(x2)));
                    for ii=ieq1
                        grad=sum(grad,ii);
                    end
                    
            end
            
            x1Minusx2_or_grad=grad;
            
    end
    