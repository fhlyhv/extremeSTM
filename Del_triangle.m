function [Jp,tri]=Del_triangle(Ltt,Lng)

% Thin membrane model based on Delaunay Triangulation
% Yu Hang, Mar.2012, NTU

p=length(Ltt);
Jp=eye(p);
tri=delaunay(Ltt,Lng);
figure; triplot(tri,Ltt,Lng); % plot the neghborhood strcture of the measuring sites

id=[Ltt,Lng];
Jp(tri(:,1)+p*(tri(:,2)-1))=-1./sum((id(tri(:,1),:)-id(tri(:,2),:)).^2,2);
Jp(tri(:,2)+p*(tri(:,1)-1))=-1./sum((id(tri(:,1),:)-id(tri(:,2),:)).^2,2);
Jp(tri(:,1)+p*(tri(:,3)-1))=-1./sum((id(tri(:,1),:)-id(tri(:,3),:)).^2,2);
Jp(tri(:,3)+p*(tri(:,1)-1))=-1./sum((id(tri(:,1),:)-id(tri(:,3),:)).^2,2);
Jp(tri(:,2)+p*(tri(:,3)-1))=-1./sum((id(tri(:,2),:)-id(tri(:,3),:)).^2,2);
Jp(tri(:,3)+p*(tri(:,2)-1))=-1./sum((id(tri(:,2),:)-id(tri(:,3),:)).^2,2);

Jp([1:p]+p*[0:p-1])=-sum(Jp)+1;

Jp=sparse(Jp);


