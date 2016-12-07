
function [F, En] =energy_atom(x0)

global nd

np=343; % 56*6+7;
nd=np*3;

aa=0.7102;
a=1.6047;
r0=2.8970;
rc=9.5;

e=exp(-a*(rc-r0));
ecut=e^2-2*e;

c=2.74412;
box1=7*c;
box2=4*sqrt(3)*c;

En=0;
fx=zeros(1,np);
fy=zeros(1,np);
fz=zeros(1,np);

%% atoms below zfix are fixed;
zfix=6.7212;

x=x0(1:np);
y=x0(np+1:2*np);
z=x0(2*np+1:3*np);

% for i=1:np
%     x(i)=x0(3*(i-1)+1);
%     y(i)=x0(3*(i-1)+2);
%     z(i)=x0(3*(i-1)+3);
% end

for i=1:np
    for j=i+1:np
        if (z(i)>zfix || z(j)>zfix)       
        %% minimum distance between i and j;
        rx=x(i)-x(j);
        rx=rx-box1.*round(rx./box1);
        ry=y(i)-y(j);
        ry=ry-box2.*round(ry./box2);
        rz=z(i)-z(j);
        r2=rx^2+ry^2+rz^2;
        
        if(r2<rc^2)
            e=exp(-a*(sqrt(r2)-r0));
        %% potential energy;
            En=En+(e^2-2*e) - ecut;

        %% force;
        ff=(e^2-e)/sqrt(r2);
        dfx=ff*rx;
        dfy=ff*ry;
        dfz=ff*rz;
        if(z(i)>zfix)
        fx(i)=fx(i)+dfx;
        fy(i)=fy(i)+dfy;
        fz(i)=fz(i)+dfz;
        end;
        if(z(j)>zfix)
        fx(j)=fx(j)-dfx;
        fy(j)=fy(j)-dfy;
        fz(j)=fz(j)-dfz;
        end;
        end
        end;

    end
end

En=En*aa;
F=[fx,fy,fz];
F=F*2*a*aa;
% plot3(fx(1:7),fy(1:7),fz(1:7),'r.', 'LineWidth',3);