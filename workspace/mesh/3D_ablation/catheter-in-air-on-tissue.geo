SetFactory("OpenCASCADE");
Merge "catheter-in-air-on-tissue.step";
Mesh.CharacteristicLengthMax = 1;
BooleanFragments{ Volume{1}; Delete; }{ Volume{2,3}; Delete; }
//+
Physical Surface("skin_side", 32) = {1, 2, 5, 4};
//+
Physical Surface("base", 33) = {3};
//+
Physical Surface("skin_top", 34) = {6};
//+
Physical Surface("air_surface", 35) = {14, 13, 12, 11, 10};
//+
Physical Volume("skin_volume", 36) = {1};
//+
Physical Volume("air_volume", 37) = {2};
//+
Physical Volume("catheter_volume", 38) = {3};
//+
Physical Surface("catheter_bottom", 41) = {7};
//+
Physical Surface("air_catheter_interface", 43) = {9, 8};
Mesh 3;

