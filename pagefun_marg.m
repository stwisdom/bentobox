function xm=pagefun_marg(x,p,dm,flag_old)

    if ~exist('flag_old','var')
        flag_old=1;
    end

    xsz=size(x);
    psz=size(p);
    
    Lxsz=length(xsz);
    Lpsz=length(psz);
    
    if Lxsz~=Lpsz
        if Lxsz>Lpsz
            if size(p,Lxsz)==1
                psz=[psz,1];
            end
        else
            if size(x,Lpsz)==1
                xsz=[xsz,1];
            end
        end
    end
    
    if dm>length(xsz)
        xsz=[xsz,ones(1,dm-length(xsz))];
        psz=[psz,ones(1,dm-length(psz))];
    end
    
    L=length(xsz);  %xsz and psz should be same length now
    xmsz=max([xsz; psz]);
    xmsz(dm)=1;
    
    if    (xsz(dm)<16 && psz(dm)<16) ...
       || isequal(xsz,psz)
       %|| ( (numel(x)>8*(max(xsz)^2) || numel(p)>8*(max(psz)^2)) && isequal(xsz,psz))
        xm=sum(bsxfun(@times,x,p),dm);
        return;
    end
    
    if flag_old
        % old way
        xr=reshape(x,[1,xsz]);
        pr=reshape(p,[1,psz]);

        xperm=[dm+1,1,setdiff(2:(1+length(xsz)),dm+1)];
        pperm=[1,dm+1,setdiff(2:(1+length(psz)),dm+1)];

        xr=permute(xr,xperm);
        pr=permute(pr,pperm);
        % clear x and p
        clear x;
        clear p;
    %     xm=pagefun(@mtimes,pr,xr);
        xm=pagefun_mtimes(pr,xr);

        clear pr;
        clear xr;

        xm=ipermute(xm,pperm);
        xmsz=size(xm);
        xm=reshape(xm,xmsz(2:end));
    
    else
        
        
        xsz=[1,xsz];
        psz=[1,psz];
        L=L+1;

        xperm=[dm+1,1,setdiff(2:L,dm+1)];
        pperm=[1,dm+1,setdiff(2:L,dm+1)];
        
        xr=permute(reshape(x,xsz),xperm);
        pr=permute(reshape(p,psz),pperm);
        % clear x and p
        clear x;
        clear p;
        
        xsz=size(xr);
        psz=size(pr);
        if length(xsz)>length(psz)
            psz=[psz,ones(1,length(xsz)-length(psz))];
        else
            xsz=[xsz,ones(1,length(psz)-length(xsz))];
        end

        psz_notdm=psz; psz_notdm(2)=1;
        [~,imaxp]=max(psz_notdm);
        pperm2=1:L;
        xperm2=1:L;
        if xsz(imaxp)==1
            pperm2(1)=imaxp;
            pperm2(imaxp)=1;
            xperm2(2)=imaxp;
            xperm2(imaxp)=2;
        else
            imaxp=[];
        end
        
        xsz_notdm=xsz; xsz_notdm(1)=1;
        [~,imaxx]=max(xsz_notdm);
        if psz(imaxx)==1 && xsz(2)==1
            xperm3=1:L;
            xperm3(2)=imaxx;
            xperm3(imaxx)=2;
        end
        
        pr=permute(pr,pperm2);
        xr=permute(xr,xperm2);
        
        if exist('xperm3','var')
            xr=permute(xr,xperm3);
        end

        xm=pagefun_mtimes(pr,xr);
        % clear xr and pr
        clear pr;
        clear xr;
        
        xm=ipermute(ipermute(xm,pperm2),pperm);
        xm=reshape(xm,xmsz);
        
%         % new way
%         i_notdm=[1:(dm-1),(dm+1):L];
%         xsz_notdm=xsz; xsz_notdm(dm)=1;
%         psz_notdm=psz; psz_notdm(dm)=1;
%         [~,imax1]=max(xsz_notdm);    %find max dimension of x
%         [~,imax2]=max(psz_notdm);    %find max dimension of p
%         imax1cand=imax1; imax2cand=imax2;
%         if imax1==imax2
%             imax1=[]; imax2=[];
%         end
%         if psz(imax1)>1 
%             imax1=[];
%         end
%         if xsz(imax2)>1
%             imax2=[];
%         end
%         i_notdm_notimax1=setdiff(i_notdm,imax1);
% 
%         if isempty(imax2)
%             imax2=1;
%         end
%         i_notdm_notimax2=setdiff(i_notdm,imax2);
% 
%         xperm=[dm,imax1,i_notdm_notimax1];
%         pperm=[imax2,dm,i_notdm_notimax2];
% 
%         x=permute(x,xperm);
%         p=permute(p,pperm);
% 
%         xm=pagefun_mtimes(p,x);
% 
%         xm=ipermute(xm,pperm);
%         xm=reshape(xm,xmsz);
    
    end

end
