cl__1 = 0.35;
cl__2 = 0.10;
Point(1) = {0, 0, 0, cl__1};
Point(2) = {1, 0, 0, cl__1};
Point(3) = {1, 1, 0, cl__2};
Point(4) = {0, 1, 0, cl__1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
//+
Physical Curve(5) = {3, 4, 1, 2};
//+
Physical Surface(6) = {1};