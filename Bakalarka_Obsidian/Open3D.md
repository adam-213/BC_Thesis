- Full resolution scans - like default config can be downsampled by as much as 99% without loosing any needed quality.
	- Adding to this - the original scan is in fp64 which is just a waste of space because the scanners are acurate to 0.5mm which is enormous in f64 scale. We can use fp32 without loosing any quality, and even fp16 without major problems - I think. This is a major help because Nvidia tensor cores use mixed precision arithmetic - fp32 and fp16. 
		- Double precision fp64 computations are used for stuff with much less tolerance than real world human scale robots.

- Poincloud from ply can be read with open source library open3d, it providdes the IO tools for loading, saving, downsamplitng, painting, cropping and other more advanced tools like normals estimation.