function aff_or_grad=bb_affine(args)

    global fxns;

    X=args.X;
    A=args.A;
    b=args.b;
    
    if isfield(args,'dim')
        dim=args.dim;
    else
        dim=[];
    end
    
    switch(nargin)
        
        case 1
            % doing a forward pass
            if isempty(dim)
                aff=fxns.pagefun_mtimes(A,X);
            else
                aff=fxns.pagefun_marg(A,X,dim);
            end
            
            aff_or_grad=aff;
            
        case 2
            % doing a backward pass
            
            aff_or_grad=grad;
    end

end