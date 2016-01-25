function idx=stridx(str,cstr)
    if iscell(str)
        Nstr=length(str);
        idx=zeros(1,Nstr);
        for nn=1:Nstr
            str_len=length(str{nn});
            idx_len_equal=cellfun(@length,cstr)==str_len;
            idx(nn)=find(idx_len_equal & strncmp(str{nn},cstr,str_len));
        end
    else
        str_len=length(str);
        idx_len_equal=cellfun(@length,cstr)==str_len;
        idx=find(idx_len_equal & strncmp(str,cstr,str_len));
    end
