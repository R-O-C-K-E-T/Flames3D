a = 2 * math.pi * t
a2 = 3 * a

flame['functions'][0]['pre_trans'][8] = math.sin(a)
flame['functions'][1]['post_trans'][3] = math.cos(a)

centre_x, centre_z = -0.472, 0.175

flame['pos'][0] = 4*math.sin(a2) + centre_x
flame['pos'][2] = 4*math.cos(a2) + centre_z

flame['rot'][1] = -a2
flame['rot'][0] = 1/3 + math.cos(a)*0.2
