function x1x2_or_grad=bb_marg(args,parentLabelGrad)

    global fxns;
    
    % extract input arguments
    x1=args.x1;
    x2=args.x2;
    dim=args.dim;   %dimension to marginalize over
    
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
    if x1sz(dim)~=x2sz(dim)
        error('Dimension %d of x1 and x2 must match!',dim);
    end
    %cut out dim, check that other dim.s are consistent with bsxfun
    x1sz=x1sz([1:dim-1,dim+1:length(x1sz)]);
    x2sz=x2sz([1:dim-1,dim+1:length(x2sz)]);
    x1sz_neq_x2sz=(x1sz~=x2sz);
    if sum(~(x1sz(x1sz_neq_x2sz)==1 | x2sz(x1sz_neq_x2sz)==1))
        error('x1 and x2 need to be compatible sizes!');
    end
    
    switch(nargin)
        
        case 1
            %%% forward pass
    
            x1x2=fxns.pagefun_marg(x1,x2,dim);
            
            x1x2_or_grad=x1x2;
            
        case 2
            %%% backward pass
            
            D_wrt_x1x2=args.D_wrt_thisNode;
            Dsz=size(D_wrt_x1x2);
            Dndims=length(Dsz);
            
            x1sz=size(x1);
            x2sz=size(x2);
            
            switch(parentLabelGrad)
  
                case 'x1'
                    if x1ndims<Dndims
                        x1sz=[x1sz,ones(1,Dndims-x1ndims)];
                    end
                    
                    ieq1=find(x1sz==1);
                    grad=bsxfun(@times,D_wrt_x1x2,x2);
                    for ii=1:ieq1
                        grad=sum(grad,ii);
                    end
                    
                case 'x2'
                    if x2ndims<Dndims
                        x2sz=[x2sz,ones(1,Dndims-x2ndims)];
                    end
                    
                    ieq1=find(x2sz==1);
                    grad=bsxfun(@times,D_wrt_x1x2,x1);
                    for ii=1:ieq1
                        grad=sum(grad,ii);
                    end
                    
            end
            
            x1x2_or_grad=grad;
            
    end
    