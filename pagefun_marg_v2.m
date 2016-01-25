function xm=pagefun_marg_v2(x1,x2,dm)


    x1sz=size(x1);
    x2sz=size(x2);
    
    L1=length(x1sz);
    L2=length(x2sz);
    
    if L1>L2
        x2sz=[x2sz,ones(1,L1-L2)];
    else
        x1sz=[x1sz,ones(1,L2-L1)];
    end
    
    xmsz=max([x1sz; x2sz]);
    xmsz(dm)=1;
    
    if dm>length(x1sz)
        xm=bsxfun(@times,x1,x2);
        return;
    end
    
    if x1sz(dm)<16 || isequal(x1sz,x2sz)
        xm=sum(bsxfun(@times,x1,x2),dm);
        return;
    end

    x1=reshape(x1,[1,x1sz]);
    x2=reshape(x2,[1,x2sz]);
    x1sz=[1,x1sz];
    x2sz=[1,x2sz];

    L=length(x1sz); %x1sz and x2sz should be same length now
    
    i_notdm=setdiff(2:L,[dm+1]);
    x1perm=[1,dm+1,i_notdm];
    x2perm=[dm+1,1,i_notdm];
    
    x1szp=x1sz(x1perm);
    x2szp=x2sz(x2perm);
    
    x1sz_notdm=x1szp; x1sz_notdm(2)=1;
    [~,imax1]=max(x1sz_notdm);
    x1perm2=1:L;
    if x2szp(imax1)==1
        x1perm2(1)=imax1;
        x1perm2(imax1)=1;
    end
    x2sz_notdm=x2szp; x2sz_notdm(1)=1;
    [~,imax2]=max(x2sz_notdm);
    x2perm2=1:L;
    if x1szp(imax2)==1
        x2perm2(2)=imax2;
        x2perm2(imax2)=2;
    end
    
    x1=permute(x1,x1perm);
    x1=permute(x1,x1perm2);
    x2=permute(x2,x2perm);
    x2=permute(x2,x2perm2);
    
    xm=pagefun_mtimes(x1,x2);
    xm=ipermute(xm,x1perm2);
    xm=ipermute(xm,x1perm);
    xm=reshape(xm,xmsz);
    
    