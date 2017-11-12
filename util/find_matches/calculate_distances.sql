select
	a.filename as a,
	b.filename as b,
	DOT_PRODUCT(a.feature_vector, b.feature_vector) as dot_dist,
	EUCLIDEAN_DISTANCE(a.feature_vector, b.feature_vector) as euc_dist
from images a inner join images b
on a.id != b.id and a.id < b.id
having euc_dist < 0.67
order by euc_dist
