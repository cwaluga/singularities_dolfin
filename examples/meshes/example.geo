cl__1 = 1;
Point(1) = {-2, -1, 0, .5};
Point(2) = {-2, 1, 0, .5};
Point(3) = {2, 1, 0, .5};
Point(4) = {2, -1, 0, .5};
Point(5) = {-1, 0.5, 0, 1};
Point(6) = {-0.5, 0.5, 0, 1};
Point(7) = {-0.5, 0, 0, 1};
Point(8) = {-1, -0, 0, 1};
Point(9) = {0.1, -0.7, 0, 1};
Point(10) = {-0.1, -0.5, 0, 1};
Point(11) = {0.1, -0.3, 0, 1};
Point(12) = {0.3, -0.5, 0, 1};
Point(13) = {0.3, 0.3, 0, 1};
Point(14) = {0.5, 0.7, 0, 1};
Point(15) = {0.7, 0.3, 0, 1};
Line(1) = {5, 8};
Line(2) = {8, 7};
Line(3) = {7, 6};
Line(4) = {6, 5};
Line(5) = {13, 14};
Line(6) = {14, 15};
Line(7) = {15, 13};
Line(8) = {10, 11};
Line(9) = {11, 12};
Line(10) = {12, 9};
Line(11) = {9, 10};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 1};
Line(15) = {1, 2};
Line Loop(20) = {12, 13, 14, 15, -3, -2, -1, -4, -7, -6, -5, -11, -10, -9, -8};
Plane Surface(20) = {20};
