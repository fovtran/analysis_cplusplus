float distanza(float4 a, float4 b)
{
	float dx, dy, dz;
 
	dx = a.x-b.x;
	dx *= dx;
 
	dy = a.y-b.y;
	dy *= dy;
 
	dz = a.z-b.z;
	dz *= dz;
 
	return dx+dy+dz;
}
 
bool isNull(float4 p)
{
	if(p.w==-1)
		return true;
 
	return false;
}
 
int findLeaf(__global float4 *model, float4 qPoint, int model_size, int cur)
{
	int best_id = -1;
	int asse;
 
	while(cur <= model_size)
	{		
		asse = model[cur].w;
 
		if(qPoint[asse] < model[cur][asse])
		{			
			if(cur*2+1>=model_size || isNull(model[cur*2+1]))
			{
				if(cur*2+2>=model_size || isNull(model[cur*2+2]))
				{
					best_id = cur;
					break;
				}
				else
				{
					cur = cur*2+2;
				}
			}
			else
			{
				cur = cur*2+1;
			}
		}
		else
		{
			if(cur*2+2>=model_size || isNull(model[cur*2+2]))
			{
				if(cur*2+1>=model_size || isNull(model[cur*2+1]))
				{
					best_id = cur;
					break;
				}
				else
				{
					cur = cur*2+1;
				}
			}
			else
			{
				cur = cur*2+2;
			}
		}
	}
 
	return best_id;
}
 
__kernel void 
nearest_neighbour(__global float4 *model,
	__global float4 *dataset,
	__global int *nearest,
	const int model_size)
{
		int g_dataset_id = get_global_id(0);
 
		float4 qPoint = dataset[g_dataset_id];
 
		int stack[7]; // 7 is enough for the number of points in my model
		int top = 0;
 
		stack[top] = -1;
 
		int node = findLeaf(model, qPoint, model_size, 0);
 
		int nn = node;
 
		int lastChild, asse, otherSide;
 
		float bestDist = distanza(qPoint, model[node]);
 
		while(node != 0)
		{
			lastChild = node;
 
			node = (node - 1) / 2;
 
			if(stack[top] == node)
			{
				--top;
			}
			else
			{
				float parentDist = distanza(qPoint, model[node]);
 
				if(parentDist < bestDist)
				{
					bestDist = parentDist;
					nn = node;
				}
 
				asse = model[node].w;
 
				float testDist = model[node][asse] - qPoint[asse];
				testDist = testDist * testDist;
 
				if(testDist < bestDist)
				{
					if (node*2+1 == lastChild)
					{
						otherSide = node*2+2;
					}
					else
					{
						otherSide = node*2+1;
					}
 
					if(otherSide < model_size && !isNull(model[otherSide]))
					{
						int testNode = findLeaf(model, qPoint, model_size, otherSide);
 
						testDist = distanza(qPoint, model[testNode]);
 
						if(testDist < bestDist)
						{
							bestDist = testDist;
							nn = testNode;
						}
 
						++top;
						stack[top] = node;
 
						node = testNode;
					}
				}
			}
		}
 
		nearest[g_dataset_id] = nn;
}