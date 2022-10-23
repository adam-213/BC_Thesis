Probably run the whole data through some sort of scaler outside the training process 
because its a lot of data.

If this is suposed to be use d on real data only thing it will have is the 3d image aka .exr files 
everything else must be treated as labels or files that labels are computed out of

its surprisingly hard to find blender files for machine part things

look into down scaling probably by 2x because then there are no problems with partial pixels and things

intensities file will probably be needed because the positions file doesn't contain enough info
	ale mohol by som ukradnut z toho positiosn file len Z height - R color a potom ho prihodit ako dalsi channel do intensities toto by drasticky zredukovalo file size - therefore loading time etc.
	I could normalize the z height to something more reasonable because realisticky it waries by like 200 - this means it could fit into fp16 with 0.1mm precision which is 5x the photoneo scanners i find it acceptable because anything more is probably just noise anyway
	but that precision only fits when between 256 and -256
	if i could squeze things into 64 to -64 i could gain additional sig fig ale podla mna to neni potrebne