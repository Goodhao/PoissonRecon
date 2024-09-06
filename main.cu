#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <stack>
#include <unordered_set>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip> // std::setprecision
#include <ctime>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

class Timer {
private:
	clock_t start_time;
	std::string label;

public:
	static void start_t(const std::string& _label = "") {
		Timer& timer = get_instance();
		timer.start_time = clock();
		timer.label = _label;
	}

	static void end_t() {
		Timer& timer = get_instance();
		clock_t end_time = clock();
		double elapsed = double(end_time - timer.start_time) / CLOCKS_PER_SEC;

		std::cout << "执行时间";
		if (!timer.label.empty()) {
			std::cout << " (" << timer.label << ")";
		}
		std::cout << ": " << elapsed << " 秒" << std::endl;
	}

private:
	Timer() {}
	static Timer& get_instance() {
		static Timer instance;
		return instance;
	}
};
#define START_T(label) Timer::start_t(label)
#define END_T() Timer::end_t()

int D = 0;

const __device__ double eps = 1e-8;
struct point {
	double x = 0, y = 0, z = 0;
	__host__ __device__ point operator+(const point& rhs) const {
		return point{ x + rhs.x,y + rhs.y,z + rhs.z };
	}
	__host__ __device__ point operator*(const double& rhs) const {
		return point{ x * rhs,y * rhs,z * rhs };
	}
	__host__ __device__ point operator/(const double& rhs) const {
		return point{ x / rhs,y / rhs,z / rhs };
	}
	__host__ __device__ point operator-(const point& rhs) const {
		return point{ x - rhs.x,y - rhs.y,z - rhs.z };
	}
	__host__ __device__ bool operator==(const point& rhs) const {
		return (fabs(x - rhs.x) < eps && fabs(y - rhs.y) < eps && fabs(z - rhs.z) < eps);
	}
};
__host__ __device__ point operator*(const double& weight, const point& p) {
	return point{ weight * p.x,weight * p.y,weight * p.z };
}

struct normal {
	double x = 0, y = 0, z = 0;
	__host__ __device__ double operator*(const normal& rhs) const {
		return x * rhs.x + y * rhs.y + z * rhs.z;
	}
	__host__ __device__ normal operator*(const double& rhs) const {
		return normal{ x * rhs,y * rhs,z * rhs };
	}
	__host__ __device__ void operator*=(const double& rhs) {
		x *= rhs;
		y *= rhs;
		z *= rhs;
	}
	__host__ __device__ normal operator+(const normal& rhs) const {
		return normal{ x + rhs.x,y + rhs.y,z + rhs.z };
	}
	__host__ __device__ normal operator/(const double& rhs) const {
		return normal{ x / rhs,y / rhs,z / rhs };
	}
	__host__ __device__ void operator/=(const double& rhs) {
		x /= rhs;
		y /= rhs;
		z /= rhs;
	}
	__host__ __device__ normal operator-(const normal& rhs) const {
		return normal{ x - rhs.x,y - rhs.y,z - rhs.z };
	}
	__host__ __device__ void operator+=(const normal& rhs) {
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
	}
};
__host__ __device__ normal operator*(const double& weight, const normal& v) {
	return normal{ weight * v.x,weight * v.y,weight * v.z };
}

int n; // 采样点个数

struct polynomial {
	int degree; // 多项式次数
	double coeffs[5]; // 系数，共degree+1个
	__host__ __device__ double operator()(double x) const {
		double res = 0;
		int i;
		for (i = 0; i <= degree; i++) {
			res += coeffs[i] * pow(x, i);
		}
		return res;
	}
	__host__ __device__ polynomial operator*(const polynomial& rhs) const {
		polynomial res;
		res.degree = degree + rhs.degree;
		int i, j, k;
		for (i = 0; i <= res.degree; i++) {
			res.coeffs[i] = 0.0;
		}
		for (i = 0; i <= degree; i++) {
			for (j = 0; j <= rhs.degree; j++) {
				k = i + j;
				res.coeffs[k] += coeffs[i] * rhs.coeffs[j];
			}
		}
		return res;
	}
	__host__ __device__ void print() {
		int i;
		for (i = 0; i <= degree; i++) {
			printf("%.9lf x^%d", coeffs[i], i);
			if (i < degree) printf(" + ");
		}
		printf("\n");
	}
};

struct function {
	// 需要用到的函数都可以用分段多项式表示
	int num = 3; // 多项式个数
	int degree = 2; // 多项式最高次数
	polynomial polys[3]; // 多项式序列，数量为num
	double break_points[4]; // 分段点序列，数量为num+1，升序排列
	__host__ __device__ double operator()(double x) const {
		int i;
		for (i = 0; i < num; i++) {
			if (break_points[i] <= x && x <= break_points[i + 1]) {
				return polys[i](x);
			}
		}
		return 0;
	}
	__host__ __device__ void print() {
		int i;
		for (i = 0; i < num; i++) {
			printf("[%.9lf, %.9lf]: ", break_points[i], break_points[i + 1]);
			polys[i].print();
		}
	}
};

__host__ __device__ double integral(const polynomial& f, double l, double r) {
	// 对多项式f在区间[l,r]上积分
	if (fabs(l - r) < 1e-15) return 0;
	double res = 0;
	int i;
	for (i = 0; i <= f.degree; i++) {
		res += f.coeffs[i] * (pow(r, i + 1) - pow(l, i + 1)) / (i + 1);
	}
	return res;
}

__host__ __device__ double integral(const function& f) {
	// 对函数f在实数轴上积分
	double res = 0;
	int i;
	for (i = 0; i < f.num; i++) {
		res += integral(f.polys[i], f.break_points[i], f.break_points[i + 1]);
	}
	return res;
}

__host__ __device__ function differential(const function& f) {
	// 我们构造的F的分段点处不一定可导，但没关系，我们求导是为了做内积，有限点处不可导不影响积分结果
	function res;
	res.num = f.num;
	if (f.degree - 1 >= 0) {
		res.degree = f.degree - 1;
	}
	else {
		res.degree = 0;
	}
	int i, j; // 如果直接在二重循环中初始化int i与int j，会导致核函数不启动。Nsight调试显示Error: cudaErrorLaunchOutOfResources(701)，这似乎说明二重循环执行时并不止用了两个寄存器空间，可能与编译器优化时的循环展开有关
	for (i = 0; i <= res.num; i++) {
		res.break_points[i] = f.break_points[i];
	}
	for (i = 0; i < f.num; i++) {
		const polynomial& p = f.polys[i];
		polynomial& p2 = res.polys[i];
		p2.degree = res.degree;
		if (p.degree > 0) {
			for (j = 0; j <= p2.degree; j++) {
				p2.coeffs[j] = p.coeffs[j + 1] * (j + 1);
			}
		}
		else {
			p2.coeffs[0] = 0;
		}
	}
	return res;
}

__host__ __device__ double inner_product(const function& a, const function& b) {
	// 负责计算两个分段多项式的函数内积
	int idx1 = 0, idx2 = 0; // idx1为当前段对应的a的多项式编号，idx2为当前段对应的b的多项式编号
	double l = 0, r = 0;
	double res = 0;
	int* idx_l;
	int* idx_r;
	while (idx1 < a.num && idx2 < b.num) {
		idx_l = nullptr;
		idx_r = nullptr;
		if (a.break_points[idx1] > b.break_points[idx2]) {
			l = a.break_points[idx1];
			idx_l = &idx2;
		}
		else {
			l = b.break_points[idx2];
			idx_l = &idx1;
		}
		if (a.break_points[idx1 + 1] < b.break_points[idx2 + 1]) {
			r = a.break_points[idx1 + 1];
			idx_r = &idx1;
		}
		else {
			r = b.break_points[idx2 + 1];
			idx_r = &idx2;
		}
		if (l > r) {
			(*idx_l)++;
			continue;
		}
		res += integral(a.polys[idx1] * b.polys[idx2], l, r);
		(*idx_r)++;
	}
	return res;
}


struct x_bits {
	unsigned int value = 0; // 32位二进制串value，表示x每一次的走向，x泛指任意一个坐标分量
	int d = 0; // value从低到高的d位是有效位
	__host__ __device__ void init(unsigned int input_value, int input_d) {
		value = input_value;
		d = input_d;
	}
	__host__ __device__ double get_center_x(double width) {
		double center_x = 0; // 因为我们总是把中心放在(0,0,0)，所以一开始坐标分量一定是0
		for (int i = 0; i < d;i++) {
			if (value & (1 << (d - 1 - i))) {
				center_x += width / 4;
			}
			else {
				center_x -= width / 4;
			}
			width /= 2;
		}
		return center_x;
	}
};

struct xyz_key {
	unsigned int branch_mask = ((1 << 3) - 1);
	unsigned int parent_mask = ~((1 << 3) - 1);
	unsigned int value = 0; // 32位二进制串value
	int d = 0; // value低3d位是有效位，从高到低写出来是x1y1z1 x2y2z2 ... xdydzd
	__host__ __device__ void init(unsigned int input_value, int input_d) {
		value = input_value;
		d = input_d;
	}
	__host__ __device__ void compute(point a[], int idx_p, int D, double width) {
		// 计算value,d
		d = D;
		point center{ 0,0,0 };
		const point& p = a[idx_p];
		value = 0;
		for (int i = 0;i < D;i++) {
			if (p.x > center.x) {
				value |= 4 << 3 * (D - 1 - i);
				center.x += width / 4;
			}
			else center.x -= width / 4;
			if (p.y > center.y) {
				value |= 2 << 3 * (D - 1 - i);
				center.y += width / 4;
			}
			else center.y -= width / 4;
			if (p.z > center.z) {
				value |= 1 << 3 * (D - 1 - i);
				center.z += width / 4;
			}
			else center.z -= width / 4;
			width /= 2;
		}
	}
	__host__ __device__ x_bits extract(int idx) {
		// idx: 0表示x 1表示y 2表示z
		int c = 0;
		if (idx == 0) c = 4;
		else if (idx == 1) c = 2;
		else c = 1;
		unsigned int res = 0;
		for (int k = 0;k < d;k++) {
			res <<= 1;
			if (value & (c << 3 * (d - 1 - k))) res |= 1;
		}
		return x_bits{ res, d };
	}
	__host__ __device__ unsigned int get_parent_key() {
		return ((value & parent_mask) >> 3);
	}
	__host__ __device__ unsigned int get_branch_key() {
		return (value & branch_mask);
	}
	__host__ __device__ bool operator<(const xyz_key& rhs) const {
		return value < rhs.value;
	}
	__host__ __device__ bool operator==(const xyz_key& rhs) const {
		return (value == rhs.value) && (d == rhs.d);
	}
	__host__ __device__ void print() {
		for (int i = 0;i < 3 * d;i++) {
			std::cout << ((value & (1 << (3 * d - 1 - i))) ? 1 : 0);
		}
		std::cout << std::endl;
	}
};

struct code {
	xyz_key key;
	int idx;
	__host__ __device__ bool operator<(const code& rhs) const {
		if (key == rhs.key) {
			return idx < rhs.idx;
		}
		else {
			return key < rhs.key;
		}
	}
};


// 三进制编码，方便枚举节点邻居
__host__ __device__ constexpr int fromTernaryChar(char c) {
	return c - '0';
}

__host__ __device__ constexpr int fromTernary(const char* ternary) {
	int result = 0;
	for (; *ternary != '\0'; ++ternary) {
		result = result * 3 + fromTernaryChar(*ternary);
	}
	return result;
}

struct node;

// 节点上的操作
struct operation {
	virtual void operator()(node* a, node* b) = 0; // 纯虚函数
};

// 八叉树节点
struct node {
	double width; // 对应立方体的宽度
	point center; // 对应立方体的中心点
	normal v; // 该节点的v（向量）
	node* parent = nullptr; // 该节点的父指针
	bool has_children = false; // 该节点是否有儿子节点
	node* children[8] = { nullptr }; // 该节点的儿子节点指针
	node* neighbors[27] = { nullptr }; // 该节点的同深度的邻居节点数组
	xyz_key key; // 节点的shuffled xyz key
	int idx_p = -1; // 该节点所含有的编号最小的采样点的编号（编号连续）
	int cnt_p = 0; // 该节点所含有的采样点的个数
	int idx_o = -1; // 该节点所含有的最底层节点在同层节点数组中的起始编号（编号连续）
	int cnt_o = 0; // 该节点所含有的最底层节点的个数
	int depth = 0; // 节点深度
	double b = 0; // 该节点的散度值
	int idx_node = 0; // 该节点的编号
	double sampling_density = 0; // 对应立方体的采样密度
	void init_children() {
		has_children = true;
		point center_child;
		int i, j, k, idx;
		for (i = 0;i < 2;i++) {
			if (i == 0) center_child.x = center.x - width / 4;
			else center_child.x = center.x + width / 4;
			for (j = 0;j < 2;j++) {
				if (j == 0) center_child.y = center.y - width / 4;
				else center_child.y = center.y + width / 4;
				for (k = 0;k < 2;k++) {
					if (k == 0) center_child.z = center.z - width / 4;
					else center_child.z = center.z + width / 4;
					idx = 4 * i + 2 * j + 1 * k;
					children[idx] = new node();
					children[idx]->center = center_child;
					children[idx]->width = width / 2;
					children[idx]->depth = depth + 1;
					children[idx]->parent = this;
					children[idx]->idx_node = -1; // 用于evaluate的时候跳过细分产生的节点，因为解向量x在这些节点上没有值
				}
			}
		}
	}
	__host__ __device__ point get_vertex(int idx) {
		point vertex = center;
		if (idx & 4) vertex.x += width / 2;
		else vertex.x -= width / 2;
		if (idx & 2) vertex.y += width / 2;
		else vertex.y -= width / 2;
		if (idx & 1) vertex.z += width / 2;
		else vertex.z -= width / 2;
		return vertex;
	}
	__host__ __device__ void compute_center() {
		// 要求已知key, width
		double root_width = width * (1 << depth);
		x_bits xbits;
		xbits = key.extract(0);
		center.x = xbits.get_center_x(root_width);
		xbits = key.extract(1);
		center.y = xbits.get_center_x(root_width);
		xbits = key.extract(2);
		center.z = xbits.get_center_x(root_width);
	}
	__device__ void naive_compute_neighbors() {
		// 要求已知parent的neighbors
		static __device__ int naive_table[27][3] = {
			{fromTernary("000"), 0b000, 0b111},
			{fromTernary("001"), 0b000, 0b110},
			{fromTernary("002"), 0b001, 0b110},
			{fromTernary("010"), 0b000, 0b101},
			{fromTernary("011"), 0b000, 0b100},
			{fromTernary("012"), 0b001, 0b100},
			{fromTernary("020"), 0b010, 0b101},
			{fromTernary("021"), 0b010, 0b100},
			{fromTernary("022"), 0b011, 0b100},
			{fromTernary("100"), 0b000, 0b011},
			{fromTernary("101"), 0b000, 0b010},
			{fromTernary("102"), 0b001, 0b010},
			{fromTernary("110"), 0b000, 0b001},
			{fromTernary("111"), 0b000, 0b000},
			{fromTernary("112"), 0b001, 0b000},
			{fromTernary("120"), 0b010, 0b001},
			{fromTernary("121"), 0b010, 0b000},
			{fromTernary("122"), 0b011, 0b000},
			{fromTernary("200"), 0b100, 0b011},
			{fromTernary("201"), 0b100, 0b010},
			{fromTernary("202"), 0b101, 0b010},
			{fromTernary("210"), 0b100, 0b001},
			{fromTernary("211"), 0b100, 0b000},
			{fromTernary("212"), 0b101, 0b000},
			{fromTernary("220"), 0b110, 0b001},
			{fromTernary("221"), 0b110, 0b000},
			{fromTernary("222"), 0b111, 0b000},
		};
		neighbors[fromTernary("111")] = this;
		if (parent == nullptr) return;
		int i, j, dir;
		for (i = 0;i < 27;i++) {
			if (parent->neighbors[i] == nullptr || !parent->neighbors[i]->has_children) {
				continue;
			}
			for (j = 0;j < 8;j++) {
				node& candidate = *parent->neighbors[i]->children[j];
				for (dir = 0;dir < 27;dir++) {
					// 测试candidate是否为当前节点dir方向上的邻居
					if (get_vertex(naive_table[dir][1]) == candidate.get_vertex(naive_table[dir][2])) {
						neighbors[dir] = &candidate;
						break;
					}
				}
			}
		}
	}
	__device__ void compute_neighbors_by_LUT() {
		// 要求已知parent的neighbors
		static __device__ int LUTparent[8][27] = {
			{0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13},
			{1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14},
			{3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16},
			{4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17},
			{9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22},
			{10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23},
			{12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25},
			{13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26}
		};
		static __device__ int LUTchild[8][27] = {
			{7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7},
			{6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6},
			{5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5},
			{4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4},
			{3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3},
			{2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2},
			{1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1},
			{0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0}
		};
		neighbors[13] = this;
		if (parent == nullptr) return;
		int idx_as_child = key.get_branch_key();
		int i;
		for (i = 0; i < 27; i++) {
			node* parent_neighbor = parent->neighbors[LUTparent[idx_as_child][i]];
			if (parent_neighbor != nullptr) {
				neighbors[i] = parent_neighbor->children[LUTchild[idx_as_child][i]];
			}
		}
	}
	void process(node* cur, operation* op) {
		// 遍历基函数与当前节点基函数有重合的所有节点，并执行相应操作
		(*op)(this, cur);
		if (cur->has_children) {
			for (int i = 0;i < 8;i++) {
				node* o = cur->children[i];
				if (o->idx_node == -1) continue;
				double dis = 1.5 * (o->width + width);
				if (fabs(o->center.x - center.x) < dis && fabs(o->center.y - center.y) < dis && fabs(o->center.z - center.z) < dis) {
					process(o, op);
				}
			}
		}
	}
};


__global__ void init_codes(code* codes, int size, point a[], int D, double width) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		new (codes + idx) code();
		codes[idx].idx = idx;
		codes[idx].key.compute(a, idx, D, width);
	}
}
__global__ void init_nodes(node* mem, node** vec_ptr, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		vec_ptr[idx] = mem + idx;
		new (vec_ptr[idx]) node();
	}
}
__global__ void update_nodes_with_samples(node** vec_ptr, node* sample_nodes, int* nodes_num, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int branch_idx = sample_nodes[i].key.get_branch_key();
		vec_ptr[(nodes_num[i] - 8) + branch_idx] = &sample_nodes[i];
	}
}
__global__ void update_nodes_without_samples(node** vec_ptr, node* sample_nodes, int* nodes_num, int d, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		if (i > 0 && nodes_num[i] == nodes_num[i - 1]) return;
		unsigned int key = (sample_nodes[i].key.get_parent_key() << 3);
		for (int j = 0; j < 8; j++) {
			node& o = *vec_ptr[(nodes_num[i] - 8) + j];
			if (o.cnt_p == 0) {
				o.depth = d;
				o.width = 1.0 / (1 << d);
				o.key.init(key + j, d);
				o.compute_center(); 
			}
		}
	}
}
__global__ void init_sample_nodes(node* sample_nodes, code* codes, node** o_of_p, int* mask, int* prefix_sum, int D, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < size) {
		if (mask[idx]) {
			new (sample_nodes + prefix_sum[idx] - 1) node();
			node& o = sample_nodes[prefix_sum[idx] - 1];
			o.depth = D;
			o.width = 1.0 / (1 << D);
			o.key = codes[idx].key;
			o.idx_p = idx;
			o.cnt_p = 1;
			o.compute_center();
			o_of_p[idx] = &o;
		}
		else {
			atomicAdd(&sample_nodes[prefix_sum[idx] - 1].cnt_p, 1);
			o_of_p[idx] = &sample_nodes[prefix_sum[idx] - 1];
		}
	}
}
__global__ void update_sample_nodes(node* sample_nodes, node** vec_ptr, int d, int size) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < size) {
		new (sample_nodes + j) node();
		node& o = sample_nodes[j];
		o.depth = d - 1;
		o.width = 1.0 / (1 << o.depth);
		o.key.init(vec_ptr[j * 8]->key.get_parent_key(), o.depth);
		o.compute_center();
		o.has_children = true;
		for (int k = 0; k < 8; k++) {
			node& u = *vec_ptr[j * 8 + k];
			o.children[k] = &u;
			u.parent = &o;
			if (u.cnt_p > 0) {
				if (o.cnt_p == 0) {
					o.idx_p = u.idx_p;
				}
				o.cnt_p += u.cnt_p;
			}
		}
	}
}
__global__ void init_nodes_num(int* nodes_num, node* sample_nodes, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		if (i == 0 || sample_nodes[i].key.get_parent_key() != sample_nodes[i - 1].key.get_parent_key()) {
			nodes_num[i] = 8;
		}
		else {
			nodes_num[i] = 0;
		}
	}
}
__global__ void set_idxs_and_neighbors(node** vec_ptr, int pre, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		vec_ptr[i]->idx_node = pre + i;
		vec_ptr[i]->compute_neighbors_by_LUT();
		//vec_ptr[i]->naive_compute_neighbors();
	}
}
__global__ void set_descendants(node** vec_ptr, int d, int D, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		if (d == D) {
			o.idx_o = i;
			o.cnt_o = 1;
		}
		else {
			if (o.has_children) {
				for (int j = 0; j < 8; j++) {
					node& o2 = *o.children[j];
					if (o.idx_o == -1) o.idx_o = o2.idx_o;
					o.cnt_o += o2.cnt_o;
				}
			}
		}
	}
}
__global__ void set_sampling_density(node** o_of_p, point a[], function* F, int D, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node* o = o_of_p[i];
		point& p = a[i];
		for (int k = D; k >= 0; k--) {
			for (int j = 0; j < 27; j++) {
				if (o->neighbors[j] == nullptr) continue;
				node& o2 = *o->neighbors[j];
				//double weight = (*F)((p.x - o2.center.x) / o2.width) * (*F)((p.y - o2.center.y) / o2.width) * (*F)((p.z - o2.center.z) / o2.width);
				//atomicAdd(&o2.sampling_density, weight);
				atomicAdd(&o2.sampling_density, 1.0); // 反而是这种简单粗暴的采样密度估计对于horse.txt来说效果最好
				// 注意：只有Compute Capability在6.0以上版本的CUDA才支持double类型的原子加法
			}
			o = o->parent;
		}
	}
}
__global__ void vector_splatting(node** o_of_p, point a[], normal va[], function* F, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *o_of_p[i];
		point& p = a[i];
		double p_sampling_density = 0, area = 0;
		for (int j = 0; j < 27; j++) {
			if (o.neighbors[j] == nullptr) continue;
			node& o2 = *o.neighbors[j];
			double weight = (*F)((p.x - o2.center.x) / o2.width) * (*F)((p.y - o2.center.y) / o2.width) * (*F)((p.z - o2.center.z) / o2.width);
			p_sampling_density += weight * o2.sampling_density;
		}
		area = 1.0 / p_sampling_density;
		normal n;
		n = va[i] * area / pow(o.width, 3); // 我不知道除以pow(o.width, 3)的用意，这里是模仿了论文1源码。因为我只把向量在最深层泼溅，所以比例系数都为1/pow(1/(1<<D), 3)，去掉后不会影响
		for (int j = 0; j < 27; j++) {
			if (o.neighbors[j] == nullptr) continue;
			node& o2 = *o.neighbors[j];
			double weight = (*F)((p.x - o2.center.x) / o2.width) * (*F)((p.y - o2.center.y) / o2.width) * (*F)((p.z - o2.center.z) / o2.width);
			atomicAdd(&o2.v.x, weight * n.x);
			atomicAdd(&o2.v.y, weight * n.y);
			atomicAdd(&o2.v.z, weight * n.z);
		}
	}
}
// 用于debug
__global__ void see(node** vec_ptr, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		printf("%d %lf\n", i, o.b);
	}
}
__global__ void set_offset(node** vec_ptr, x_bits* offset[], int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		for (int k = 0; k < 3; k++) {
			offset[k][o.idx_node] = o.key.extract(k);
		}
	}
}
__global__ void compute_mask(int* mask, code* codes, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		if (i == 0 || !(codes[i].key == codes[i - 1].key)) {
			mask[i] = 1;
		}
		else {
			mask[i] = 0;
		}
	}
}
struct is_true {
	__host__ __device__ bool operator()(bool x) {
		return x;
	}
};
__global__ void set_functions(function* key_to_Fo, function* F, int d, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		x_bits xbits;
		xbits.init(i, d);
		double x = xbits.get_center_x(1.0);
		double width = 1.0 / (1 << d);
		function& Fo = key_to_Fo[i];
		Fo.num = 3;
		Fo.degree = 2;
		for (int i = 0; i < 3; i++) {
			Fo.polys[i].degree = 2;
			Fo.polys[i].coeffs[0] = F->polys[i].coeffs[0] - F->polys[i].coeffs[1] * x / width + F->polys[i].coeffs[2] * x * x / (width * width);
			Fo.polys[i].coeffs[1] = F->polys[i].coeffs[1] / width - F->polys[i].coeffs[2] * 2 * x / (width * width);
			Fo.polys[i].coeffs[2] = F->polys[i].coeffs[2] / (width * width);
		}
		for (int i = 0; i < 4; i++) Fo.break_points[i] = F->break_points[i] * width + x;
	}
}
__global__ void set_inner_product_table(function* key_to_Fo1, function* key_to_Fo2, double* table1, double* table2, double* table3, int len2, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int key1 = i / len2;
		int key2 = i % len2;
		function Fo1 = key_to_Fo1[key1];
		function Fo2 = key_to_Fo2[key2];
		function dFo1 = differential(Fo1);
		function ddFo1 = differential(dFo1);
		table1[key1 * len2 + key2] = inner_product(Fo1, Fo2);
		table2[key1 * len2 + key2] = inner_product(dFo1, Fo2);
		table3[key1 * len2 + key2] = inner_product(ddFo1, Fo2);
	}
}
__global__ void compute_divergence(int D, node** vec_ptr, x_bits* offset[], double** table1, double** table2, double** table3, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node* o = vec_ptr[i];
		node* o2 = o;
		for (int d = D; d >= 0; d--) {
			for (int j = 0; j < 27; j++) {
				if (o2->neighbors[j] == nullptr) continue;
				node* o3 = o2->neighbors[j];
				normal u;
				u.x = table2[d][offset[0][o->idx_node].value * (1 << d) + offset[0][o3->idx_node].value] * table1[d][offset[1][o->idx_node].value * (1 << d) + offset[1][o3->idx_node].value] * table1[d][offset[2][o->idx_node].value * (1 << d) + offset[2][o3->idx_node].value];
				u.y = table2[d][offset[1][o->idx_node].value * (1 << d) + offset[1][o3->idx_node].value] * table1[d][offset[0][o->idx_node].value * (1 << d) + offset[0][o3->idx_node].value] * table1[d][offset[2][o->idx_node].value * (1 << d) + offset[2][o3->idx_node].value];
				u.z = table2[d][offset[2][o->idx_node].value * (1 << d) + offset[2][o3->idx_node].value] * table1[d][offset[0][o->idx_node].value * (1 << d) + offset[0][o3->idx_node].value] * table1[d][offset[1][o->idx_node].value * (1 << d) + offset[1][o3->idx_node].value];
				u *= -1;
				atomicAdd(&o3->b, o->v * u);
			}
			o2 = o2->parent;
		}
	}
}
__global__ void prepare_CSR(node** vec_ptr, int d, int d2, int* csrOffsets, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		node* pa = &o;
		for (int j = 0; j < d - d2; j++) pa = pa->parent;
		int sz = 0;
		for (int j = 0; j < 27; j++) {
			if (pa->neighbors[j] != nullptr) {
				++sz;
			}
		}
		csrOffsets[i + 1] = sz;
	}
}
__global__ void set_b(node** vec_ptr, float* b, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		b[i] = vec_ptr[i]->b;
	}
}
__global__ void construct_laplacian_matrix(node** vec_ptr, int d, int d2, int pre_size, int pre_size2, x_bits* offset[], double* table1, double* table2, double* table3, int* csrOffsets, int* columns, float* values, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		int idx = o.idx_node - pre_size;
		node* pa = &o;
		for (int j = 0; j < d - d2; j++) pa = pa->parent;
		int sz = 0;
		for (int j = 0; j < 27; j++) {
			if (pa->neighbors[j] != nullptr) {
				node& o2 = *pa->neighbors[j];
				int idx2 = o2.idx_node - pre_size2;
				double value = 0;
				value += table3[offset[0][o.idx_node].value * (1 << d2) + offset[0][o2.idx_node].value] * table1[offset[1][o.idx_node].value * (1 << d2) + offset[1][o2.idx_node].value] * table1[offset[2][o.idx_node].value * (1 << d2) + offset[2][o2.idx_node].value];
				value += table3[offset[1][o.idx_node].value * (1 << d2) + offset[1][o2.idx_node].value] * table1[offset[0][o.idx_node].value * (1 << d2) + offset[0][o2.idx_node].value] * table1[offset[2][o.idx_node].value * (1 << d2) + offset[2][o2.idx_node].value];
				value += table3[offset[2][o.idx_node].value * (1 << d2) + offset[2][o2.idx_node].value] * table1[offset[1][o.idx_node].value * (1 << d2) + offset[1][o2.idx_node].value] * table1[offset[0][o.idx_node].value * (1 << d2) + offset[0][o2.idx_node].value];
				value = -value;
				columns[csrOffsets[i] + sz] = idx2;
				values[csrOffsets[i] + sz] = value;
				++sz;
			}
		}
	}
}
__global__ void set_value_table(function* key_to_Fo, int d, double* table4, double width, int num, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		unsigned int key = i / num;
		int j = i % num;
		function Fo = key_to_Fo[key];
		table4[key * num + j] = Fo(j * width - 0.5);
	}
}
__device__ double evaluate_point(int D, node* cur, point p, unsigned int index_x, unsigned int index_y, unsigned int index_z, double* table4[], x_bits* offset[], float* solution) {
	int num = (1 << (D + 1)) + 1;
	unsigned int index = (index_x << 2 * (D + 2)) + (index_y << (D + 2)) + index_z;
	int h = -1;
	node* s[8 * 10];
	s[++h] = cur;
	double weight;
	double value = 0;
	int i;
	while (h != -1) {
		cur = s[h--];
		weight = table4[cur->depth][offset[0][cur->idx_node].value * num + index_x] * table4[cur->depth][offset[1][cur->idx_node].value * num + index_y] * table4[cur->depth][offset[2][cur->idx_node].value * num + index_z];
		value += weight * solution[cur->idx_node];
		if (cur->has_children) {
			for (i = 0; i < 8; i++) {
				node& o = *cur->children[i];
				//if (o.idx_node != -1 && fabs(p.x - o.center.x) < 1.5 * o.width && fabs(p.y - o.center.y) < 1.5 * o.width && fabs(p.z - o.center.z) < 1.5 * o.width) {
				if (o.idx_node != -1 && fabs(p.x - o.center.x) < o.width && fabs(p.y - o.center.y) < o.width && fabs(p.z - o.center.z) < o.width) {
					s[++h] = &o;
				}
			}
		}
	}
	return value;
}
__global__ void compute_iso_value(node** vec_ptr, double* sum, double* val, int D, node* root, double* table4[], x_bits* offset[], float* solution, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		double w = sqrt(o.v * o.v);
		unsigned int index_x, index_y, index_z, index;
		point& p = o.center;
		index_x = round((p.x + 0.5) * (1 << (D + 1)));
		index_y = round((p.y + 0.5) * (1 << (D + 1)));
		index_z = round((p.z + 0.5) * (1 << (D + 1)));
		/*(*val) += w * evaluate_point(D, root, p, index_x, index_y, index_z, table4, offset, solution);
		(*sum) += w;*/
		atomicAdd(val, w * evaluate_point(D, root, p, index_x, index_y, index_z, table4, offset, solution)); // val, sum已经是指针，就不用再取地址了
		atomicAdd(sum, w);
	}
}
__device__ unsigned int get_index(const point& p, double width, int D) {
	// 注意这个函数与CPU版本不同，因为GPU版本我们只保留了对最深层节点中心点的get_index，分辨率可以粗一点
	// width是最细节点的宽度
	unsigned int index_x, index_y, index_z, index;
	index_x = round((p.x + 0.5 - 0.5 * width) / width);
	index_y = round((p.y + 0.5 - 0.5 * width) / width);
	index_z = round((p.z + 0.5 - 0.5 * width) / width);
	index = (index_x << 2 * D) + (index_y << D) + index_z;
	return index;
}
__global__ void put(node** vec_ptr, point* arr, int* vis, double width, int D, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		node& o = *vec_ptr[i];
		arr[i] = o.center;
		vis[get_index(o.center, width, D)] = 1;
	}
}
__global__ void evaluate_vertices(int D, point* arr, double* vals, int* d_sz, double width, node* root, double* table4[], x_bits* offset[], float* solution, double iso_value) {
	static __device__ point pp[8] = {
		point{ -0.5, -0.5, +0.5 },
		point{ -0.5, +0.5, +0.5 },
		point{ -0.5, +0.5, -0.5 },
		point{ -0.5, -0.5, -0.5 },
		point{ +0.5, -0.5, +0.5 },
		point{ +0.5, +0.5, +0.5 },
		point{ +0.5, +0.5, -0.5 },
		point{ +0.5, -0.5, -0.5 }
	};
	int size = *d_sz * 8;
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < size) {
		int i = u / 8;
		int j = u % 8;
		point p = arr[i] + pp[j] * width;
		unsigned int index_x, index_y, index_z, index;
		index_x = round((p.x + 0.5) * (1 << (D + 1)));
		index_y = round((p.y + 0.5) * (1 << (D + 1)));
		index_z = round((p.z + 0.5) * (1 << (D + 1)));
		vals[i * 8 + j] = evaluate_point(D, root, p, index_x, index_y, index_z, table4, offset, solution) - iso_value;
	}
}
__global__ void extend(int D, double* vals, point* arr1, point* arr2, int* d_sz1, int* d_sz2, int* vis, int* has_root, double width) {
	static __device__ int edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	static __device__ point pp[8] = {
		point{ -0.5, -0.5, +0.5 },
		point{ -0.5, +0.5, +0.5 },
		point{ -0.5, +0.5, -0.5 },
		point{ -0.5, -0.5, -0.5 },
		point{ +0.5, -0.5, +0.5 },
		point{ +0.5, +0.5, +0.5 },
		point{ +0.5, +0.5, -0.5 },
		point{ +0.5, -0.5, -0.5 }
	};
	int size = *d_sz1 * 12;
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	if (u < size) {
		int i = u / 12;
		int e = u % 12;
		point& node_center = arr1[i];
		point new_node_center;
		unsigned int index_x, index_y, index_z, index;
		double v1, v2;
		int idx1, idx2;
		point center;
		point p1, p2;
		int dir;
		int idx;
		int old;
		idx1 = edges[e][0];
		idx2 = edges[e][1];
		p1 = node_center + pp[idx1] * width;
		p2 = node_center + pp[idx2] * width;
		center = (p1 + p2) / 2;
		dir = 0;
		if (e == 8 || e == 9 || e == 10 || e == 11) dir = 0; // x轴方向
		if (e == 0 || e == 2 || e == 4 || e == 6) dir = 1; // y轴方向
		if (e == 1 || e == 3 || e == 5 || e == 7) dir = 2; // z轴方向
		v1 = vals[i * 8 + idx1];
		v2 = vals[i * 8 + idx2];
		if (v1 < 0 && v2 > 0 || v1 > 0 && v2 < 0) {
			// 枚举边的3个虚拟邻居
			new_node_center = 2 * (center - node_center) + node_center;
			index = get_index(new_node_center, width, D);
			old = atomicCAS(&vis[index], 0, 1);
			if (old == 0) {
				idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center; // atomicAdd(d_sz2, 1)返回的是旧值，等效于arr2[h++]=new_node_center
			}
			if (dir == 0) {
				new_node_center = { node_center.x, node_center.y, 2 * (center.z - node_center.z) + node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
				new_node_center = { node_center.x, 2 * (center.y - node_center.y) + node_center.y, node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
			}
			else if (dir == 1) {
				new_node_center = { node_center.x, node_center.y, 2 * (center.z - node_center.z) + node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
				new_node_center = { 2 * (center.x - node_center.x) + node_center.x, node_center.y, node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
			}
			else if (dir == 2) {
				new_node_center = { node_center.x, 2 * (center.y - node_center.y) + node_center.y, node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
				new_node_center = { 2 * (center.x - node_center.x) + node_center.x, node_center.y, node_center.z };
				index = get_index(new_node_center, width, D);
				old = atomicCAS(&vis[index], 0, 1);
				if (old == 0) {
					idx = atomicAdd(d_sz2, 1); arr2[idx] = new_node_center;
				}
			}
			atomicExch(&has_root[get_index(node_center, width, D)], 1);
		}
	}
}
__global__ void collect_useful_nodes(point* arr, double* vals, point* res, double* res_vals, int* d_sz, int* d_res_sz, int* has_root, int D, double width) {
	int size = *d_sz;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		point& node_center = arr[i];
		int j;
		if (has_root[get_index(node_center, width, D)]) {
			int idx = atomicAdd(d_res_sz, 1);
			res[idx] = node_center;
			for (j = 0; j < 8; j++) {
				res_vals[idx * 8 + j] = vals[i * 8 + j];
			}
		}
	}
}
struct triangle {
	point p[3];
};
__global__ void marching_cube(double width, point* res, double* res_vals, int* d_res_sz, triangle* triangles, int* d_triangles_sz) {
	static __device__ int marching_cube_table[256][15] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
	};
	int size = *d_res_sz;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		int j, k, e;
		int v[2];
		double k0, k1;
		double v0, v1;
		point p[8];
		p[0] = res[i] + point{ -width / 2, -width / 2, +width / 2 };
		p[1] = res[i] + point{ -width / 2, +width / 2, +width / 2 };
		p[2] = res[i] + point{ -width / 2, +width / 2, -width / 2 };
		p[3] = res[i] + point{ -width / 2, -width / 2, -width / 2 };
		p[4] = res[i] + point{ +width / 2, -width / 2, +width / 2 };
		p[5] = res[i] + point{ +width / 2, +width / 2, +width / 2 };
		p[6] = res[i] + point{ +width / 2, +width / 2, -width / 2 };
		p[7] = res[i] + point{ +width / 2, -width / 2, -width / 2 };
		int idx = 0;
		for (j = 7; j >= 0; j--) {
			idx <<= 1;
			idx += (res_vals[i * 8 + j] > 0);
		}
		triangle t;
		for (j = 0; j < 5; j++) {
			bool computed = false;
			for (k = 0; k < 3; k++) {
				e = marching_cube_table[idx][j * 3 + k];
				if (e == -1) {
					computed = false;
					break;
				}
				if (e == 0) {
					v[0] = 0;
					v[1] = 1;
				}
				else if (e == 1) {
					v[0] = 1;
					v[1] = 2;
				}
				else if (e == 2) {
					v[0] = 2;
					v[1] = 3;
				}
				else if (e == 3) {
					v[0] = 3;
					v[1] = 0;
				}
				else if (e == 4) {
					v[0] = 4;
					v[1] = 5;
				}
				else if (e == 5) {
					v[0] = 5;
					v[1] = 6;
				}
				else if (e == 6) {
					v[0] = 6;
					v[1] = 7;
				}
				else if (e == 7) {
					v[0] = 7;
					v[1] = 4;
				}
				else if (e == 8) {
					v[0] = 0;
					v[1] = 4;
				}
				else if (e == 9) {
					v[0] = 1;
					v[1] = 5;
				}
				else if (e == 10) {
					v[0] = 2;
					v[1] = 6;
				}
				else if (e == 11) {
					v[0] = 3;
					v[1] = 7;
				}
				v0 = res_vals[i * 8 + v[0]];
				v1 = res_vals[i * 8 + v[1]];
				assert(v0 > 0 && v1 < 0 || v0 < 0 && v1 > 0);
				k0 = -v1 / (v0 - v1);
				k1 = v0 / (v0 - v1);
				assert(0 <= k0 && k0 <= 1);
				assert(0 <= k1 && k1 <= 1);
				t.p[k] = p[v[0]] * k0 + p[v[1]] * k1;
				computed = true;
			}
			if (computed) {
				int h = atomicAdd(d_triangles_sz, 1);
				triangles[h] = t;
			}
		}
	}
}
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
int SparseMatrixVectorMultiply(
	int numRows, int numCols, int nnz,
	const int* d_csrOffsets, const int* d_columns, const float* d_values,
	const float* dX, float* dY, float alpha, float beta)
{
	cusparseHandle_t handle;
	cusparseSpMatDescr_t matA;
	cusparseDnVecDescr_t vecX, vecY;
	size_t bufferSize = 0;
	void* dBuffer = nullptr;

	// 创建 cuSPARSE handle
	CHECK_CUSPARSE(cusparseCreate(&handle));

	// 创建稀疏矩阵描述符
	CHECK_CUSPARSE(cusparseCreateCsr(
		&matA, numRows, numCols, nnz,
		(void*)d_csrOffsets, (void*)d_columns, (void*)d_values,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

	// 创建稠密向量描述符
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, numCols, (void*)dX, CUDA_R_32F));
	CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, numRows, (void*)dY, CUDA_R_32F));

	// 检查 SpMV 缓冲区的大小并分配内存
	CHECK_CUSPARSE(cusparseSpMV_bufferSize(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));


	CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

	// 执行 SpMV 运算
	CHECK_CUSPARSE(cusparseSpMV(
		handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
		&alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
		CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
	
	CHECK_CUDA(cudaFree(dBuffer));

	CHECK_CUSPARSE(cusparseDestroySpMat(matA));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
	CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
	CHECK_CUSPARSE(cusparseDestroy(handle));

	return EXIT_SUCCESS;
}
__host__ int solve_conjugate_gradient(int* d_csrOffsets, int* d_columns, float* d_values, float* d_b, float* d_x, int len, int iters) {
	const float eps = 1e-6;
	float alpha, beta, r_dot_r, b_dot_b;
	float neg_alpha;
	float one = 1.0f;
	float* h_x = (float*)malloc(len * sizeof(float));
	for (int i = 0; i < len; i++) h_x[i] = 0;
	cudaMemcpy(d_x, h_x, len * sizeof(float), cudaMemcpyHostToDevice);
	float* d_d;
	float* d_r;
	cudaMalloc(&d_d, len * sizeof(float));
	cudaMalloc(&d_r, len * sizeof(float));
	cudaMemcpy(d_d, d_b, len * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_r, d_b, len * sizeof(float), cudaMemcpyDeviceToDevice);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSdot(handle, len, d_r, 1, d_r, 1, &r_dot_r);
	cublasSdot(handle, len, d_b, 1, d_b, 1, &b_dot_b);
	
	if (b_dot_b >= eps) {
		float* d_Md;
		cudaMalloc(&d_Md, len * sizeof(float));
		int num_rows = len;
		int num_cols = len;
		int nnz;
		cudaMemcpy(&nnz, d_csrOffsets + len, sizeof(int), cudaMemcpyDeviceToHost);
		float temp;
		for (int i = 0; i < iters; i++) {
			SparseMatrixVectorMultiply(num_rows, num_cols, nnz, d_csrOffsets, d_columns, d_values, d_d, d_Md, 1.0f, 0.0f);
			cublasSdot(handle, len, d_d, 1, d_Md, 1, &temp);
			if (fabs(temp) < eps) break;
			alpha = r_dot_r / temp;
			neg_alpha = -alpha;
			cublasSaxpy(handle, len, &neg_alpha, d_Md, 1, d_r, 1);
			cublasSdot(handle, len, d_r, 1, d_r, 1, &temp);
			if (temp / b_dot_b < eps) break;
			beta = temp / r_dot_r;
			cublasSaxpy(handle, len, &alpha, d_d, 1, d_x, 1);
			if (beta < eps) break;
			r_dot_r = temp;
			cublasSscal(handle, len, &beta, d_d, 1);
			cublasSaxpy(handle, len, &one, d_r, 1, d_d, 1);
		}
		cudaFree(d_Md);
	}
	cublasDestroy(handle);
	cudaFree(d_d);
	cudaFree(d_r);
	free(h_x);
	return 0;
}
int main() {
	// 论文1：Poisson Surface Reconstruction
	// 论文2：Data-Parallel Octrees for Surface Reconstruction

	/*cudaDeviceProp prop;
	int device;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);*/


	// 第一步：生成八叉树
	std::ifstream point_cloud;
	point_cloud.open("armadillo点云.txt");
	//point_cloud.open("horse.txt");
	//point_cloud.open("sphere_points_and_normals.txt");
	//point_cloud.open("cube.txt");
	//point_cloud.open("plane_points_and_normals.txt");
	//point_cloud.open("small_plane_points_and_normals.txt");

	point_cloud >> n;

	point* h_a;
	point* h_sorted_a;
	point* d_a;
	normal* h_va;
	normal* h_sorted_va;
	normal* d_va;
	h_a = (point*)malloc(n * sizeof(point));
	h_va = (normal*)malloc(n * sizeof(normal));
	h_sorted_a = (point*)malloc(n * sizeof(point));
	h_sorted_va = (normal*)malloc(n * sizeof(normal));
	cudaMalloc(&d_a, n * sizeof(point));
	cudaMalloc(&d_va, n * sizeof(normal));


	double min_x = 1e10, max_x = -1e10, min_y = 1e10, max_y = -1e10, min_z = 1e10, max_z = -1e10;
	for (int i = 0; i < n; i++) {
		double x, y, z, vx, vy, vz;
		point_cloud >> x >> y >> z >> vx >> vy >> vz;
		h_a[i] = point{ x,y,z };
		h_va[i] = normal{ vx,vy,vz };
		min_x = std::min(min_x, x);
		min_y = std::min(min_y, y);
		min_z = std::min(min_z, z);
		max_x = std::max(max_x, x);
		max_y = std::max(max_y, y);
		max_z = std::max(max_z, z);
	}
	point_cloud.close();
	point center;
	center.x = (min_x + max_x) / 2;
	center.y = (min_y + max_y) / 2;
	center.z = (min_z + max_z) / 2;
	D = 8;
	double scale_ratio = 1.25;
	double width = std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z));
	for (int i = 0; i < n; i++) {
		h_a[i].x -= center.x;
		h_a[i].y -= center.y;
		h_a[i].z -= center.z;
	}
	for (int i = 0; i < n; i++) {
		h_a[i].x /= width * scale_ratio;
		h_a[i].y /= width * scale_ratio;
		h_a[i].z /= width * scale_ratio;
		assert(-0.5 <= h_a[i].x && h_a[i].x <= 0.5);
		assert(-0.5 <= h_a[i].y && h_a[i].y <= 0.5);
		assert(-0.5 <= h_a[i].z && h_a[i].z <= 0.5);
	}
	width = 1;
	center.x = center.y = center.z = 0; // 把中心放在(0,0,0)处

	cudaMemcpy(d_a, h_a, n * sizeof(point), cudaMemcpyHostToDevice);
	cudaMemcpy(d_va, h_va, n * sizeof(normal), cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024;
	int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

	START_T("计算codes");
	code* h_codes;
	code* d_codes;
	h_codes = (code*)malloc(n * sizeof(code));
	cudaMalloc(&d_codes, n * sizeof(code));
	init_codes << <numBlocks, threadsPerBlock >> > (d_codes, n, d_a, D, width);
	cudaDeviceSynchronize();
	END_T();

	START_T("排序codes");
	thrust::device_ptr<code> d_ptr(d_codes);
	thrust::sort(d_ptr, d_ptr + n);
	cudaDeviceSynchronize();
	END_T();

	START_T("CUDA复制");
	cudaMemcpy(h_codes, d_codes, n * sizeof(code), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_a, d_a, n * sizeof(point), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_va, d_va, n * sizeof(normal), cudaMemcpyDeviceToHost);
	for (int i = 0; i < n; i++) {
		int idx = h_codes[i].idx;
		h_sorted_a[i] = h_a[idx];
		h_sorted_va[i] = h_va[idx];
	}
	cudaDeviceSynchronize();
	cudaMemcpy(d_a, h_sorted_a, n * sizeof(point), cudaMemcpyHostToDevice);
	cudaMemcpy(d_va, h_sorted_va, n * sizeof(normal), cudaMemcpyHostToDevice);
	END_T();

	START_T("去重");
	int* d_mask;
	int* d_prefix_sum;
	cudaMalloc(&d_mask, n * sizeof(int));
	cudaMalloc(&d_prefix_sum, n * sizeof(int));
	compute_mask << <numBlocks, threadsPerBlock >> > (d_mask, d_codes, n);
	cudaDeviceSynchronize();
	thrust::device_ptr<int> d_ptr2(d_mask);
	thrust::device_ptr<int> d_ptr3(d_prefix_sum);
	thrust::inclusive_scan(thrust::device, d_ptr2, d_ptr2 + n, d_ptr3);
	cudaDeviceSynchronize();
	int m;
	cudaMemcpy(&m, &d_prefix_sum[n - 1], sizeof(int), cudaMemcpyDeviceToHost);
	END_T();

	START_T("构造含有采样点的节点");
	node* d_sample_nodes;
	node** d_o_of_p;
	cudaMalloc(&d_sample_nodes, n * sizeof(node));
	cudaMalloc(&d_o_of_p, n * sizeof(node*));
	init_sample_nodes << <numBlocks, threadsPerBlock >> > (d_sample_nodes, d_codes, d_o_of_p, d_mask, d_prefix_sum, D, n);
	cudaDeviceSynchronize();
	END_T();
	free(h_codes);
	cudaFree(d_codes);

	START_T("构造nodes_num");
	int* d_nodes_num_before;
	int* d_nodes_num;
	cudaMalloc(&d_nodes_num_before, n * sizeof(int));
	cudaMalloc(&d_nodes_num, n * sizeof(int));
	init_nodes_num << <numBlocks, threadsPerBlock >> > (d_nodes_num_before, d_sample_nodes, m);
	cudaDeviceSynchronize();

	thrust::device_ptr<int> d_ptr4(d_nodes_num_before);
	thrust::device_ptr<int> d_ptr5(d_nodes_num);
	thrust::inclusive_scan(thrust::device, d_ptr4, d_ptr4 + m, d_ptr5);
	cudaDeviceSynchronize();
	int tot;
	cudaMemcpy(&tot, &d_nodes_num[m - 1], sizeof(int), cudaMemcpyDeviceToHost);
	END_T();

	START_T("初始化nodes_d[D]");
	int h_size_d[10];
	h_size_d[D] = tot;
	thrust::device_vector<node*> d_nodes_d[10];
	d_nodes_d[D].resize(tot);
	node** vec_ptr = thrust::raw_pointer_cast(d_nodes_d[D].data());
	node* mem;
	cudaMalloc(&mem, tot * sizeof(node));
	init_nodes << <numBlocks, threadsPerBlock >> > (mem, vec_ptr, tot);
	cudaDeviceSynchronize();
	END_T();

	START_T("更新nodes_d[D]");
	update_nodes_with_samples << <numBlocks, threadsPerBlock >> > (vec_ptr, d_sample_nodes, d_nodes_num, m);
	cudaDeviceSynchronize();
	update_nodes_without_samples << <numBlocks, threadsPerBlock >> > (vec_ptr, d_sample_nodes, d_nodes_num, D, m);
	cudaDeviceSynchronize();
	END_T();
	START_T("递推构造所有nodes_d[i]");
	for (int i = D; i >= 1; i--) {
		// 已知nodes_d[i]，去构建nodes_d[i-1]
		m = tot / 8;
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[i].data());
		cudaMalloc(&d_sample_nodes, m * sizeof(node));
		update_sample_nodes << <numBlocks, threadsPerBlock >> > (d_sample_nodes, vec_ptr, i, m);
		cudaDeviceSynchronize();
		init_nodes_num << <numBlocks, threadsPerBlock >> > (d_nodes_num_before, d_sample_nodes, m);
		cudaDeviceSynchronize();
		thrust::device_ptr<int> d_ptr4(d_nodes_num_before);
		thrust::device_ptr<int> d_ptr5(d_nodes_num);
		thrust::inclusive_scan(thrust::device, d_ptr4, d_ptr4 + m, d_ptr5);
		cudaDeviceSynchronize();
		cudaMemcpy(&tot, &d_nodes_num[m - 1], sizeof(int), cudaMemcpyDeviceToHost);
		if (i == 1) tot = 1;
		std::cout << i << " " << tot << std::endl;
		h_size_d[i - 1] = tot;
		d_nodes_d[i - 1].resize(tot);
		cudaMalloc(&mem, tot * sizeof(node));
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[i - 1].data());
		init_nodes << <numBlocks, threadsPerBlock >> > (mem, vec_ptr, tot);
		cudaDeviceSynchronize();
		update_nodes_with_samples << <numBlocks, threadsPerBlock >> > (vec_ptr, d_sample_nodes, d_nodes_num, m);
		cudaDeviceSynchronize();
		if (i > 1) {
			update_nodes_without_samples << <numBlocks, threadsPerBlock >> > (vec_ptr, d_sample_nodes, d_nodes_num, i - 1, m);
			cudaDeviceSynchronize();
		}
	}
	END_T();

	START_T("给节点编号并计算邻居");
	int tot_size = 0;
	for (int d = 0; d <= D; d++) {
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[d].data());
		set_idxs_and_neighbors << <numBlocks, threadsPerBlock >> > (vec_ptr, tot_size, h_size_d[d]);
		cudaDeviceSynchronize();
		tot_size += h_size_d[d];
	}
	END_T();
	START_T("计算最深层后代区间");
	for (int d = D; d >= 0; d--) {
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[d].data());
		set_descendants << <numBlocks, threadsPerBlock >> > (vec_ptr, d, D, h_size_d[d]);
		cudaDeviceSynchronize();
	}
	END_T();

	START_T("计算坐标各分量的编码");
	x_bits** d_offset; // 这里更换了CPU版本的索引顺序，维度索引为第一索引，这样malloc次数少一点
	cudaMalloc(&d_offset, 3 * sizeof(x_bits*));
	x_bits* h_offset[3];
	for (int i = 0; i < 3; i++) {
		cudaMalloc(&h_offset[i], tot_size * sizeof(x_bits));  // 为每个指针分配设备内存
	}
	cudaMemcpy(d_offset, h_offset, 3 * sizeof(x_bits*), cudaMemcpyHostToDevice);
	for (int d = 0; d <= D; d++) {
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[d].data());
		set_offset << <numBlocks, threadsPerBlock >> > (vec_ptr, d_offset, h_size_d[d]);
		cudaDeviceSynchronize();
	}
	END_T();
	std::cout << "节点个数：" << tot_size << std::endl;

	START_T("构造标准基函数");
	function h_F;
	h_F.polys[0].degree = 2;
	h_F.polys[1].degree = 2;
	h_F.polys[2].degree = 2;
	h_F.polys[0].coeffs[0] = 1.125;
	h_F.polys[0].coeffs[1] = 1.5;
	h_F.polys[0].coeffs[2] = 0.5;
	h_F.polys[1].coeffs[0] = 0.75;
	h_F.polys[1].coeffs[1] = 0;
	h_F.polys[1].coeffs[2] = -1;
	h_F.polys[2].coeffs[0] = 1.125;
	h_F.polys[2].coeffs[1] = -1.5;
	h_F.polys[2].coeffs[2] = 0.5;
	h_F.break_points[0] = -1.5;
	h_F.break_points[1] = -0.5;
	h_F.break_points[2] = 0.5;
	h_F.break_points[3] = 1.5;
	function* d_F;
	cudaMalloc(&d_F, sizeof(function));
	cudaMemcpy(d_F, &h_F, sizeof(function), cudaMemcpyHostToDevice);
	END_T();

	START_T("构造向量场");
	set_sampling_density << <numBlocks, threadsPerBlock >> > (d_o_of_p, d_a, d_F, D, n);
	cudaDeviceSynchronize();
	vector_splatting << <numBlocks, threadsPerBlock >> > (d_o_of_p, d_a, d_va, d_F, n);
	cudaDeviceSynchronize();
	END_T();
	/*vec_ptr = thrust::raw_pointer_cast(d_nodes_d[D].data());
	see << <numBlocks, threadsPerBlock >> > (vec_ptr, h_size_d[D]);
	cudaDeviceSynchronize();*/

	START_T("构造节点函数");
	function* key_to_Fo[10];
	for (int d = 0; d <= D; d++) {
		cudaMalloc(&key_to_Fo[d], (1 << d) * sizeof(function));
		set_functions << <numBlocks, threadsPerBlock >> > (key_to_Fo[d], d_F, d, 1 << d);
		cudaDeviceSynchronize();
	}
	END_T();

	START_T("构造函数内积表");
	double* table1[10][10];
	double* table2[10][10];
	double* table3[10][10];
	for (int d1 = 0; d1 <= D; d1++) {
		for (int d2 = 0; d2 <= D; d2++) {
			int len1 = (1 << d1), len2 = (1 << d2);
			cudaMalloc(&table1[d1][d2], len1 * len2 * sizeof(double));
			cudaMalloc(&table2[d1][d2], len1 * len2 * sizeof(double));
			cudaMalloc(&table3[d1][d2], len1 * len2 * sizeof(double));
			cudaDeviceSynchronize();
			int threadsPerBlock = 1024;
			int numBlocks = (len1 * len2 + threadsPerBlock - 1) / threadsPerBlock;
			set_inner_product_table << <numBlocks, threadsPerBlock >> > (key_to_Fo[d1], key_to_Fo[d2], table1[d1][d2], table2[d1][d2], table3[d1][d2], len2, len1 * len2);
			cudaDeviceSynchronize();
		}
	}
	END_T();

	START_T("计算散度");
	double** ttable1;
	double** ttable2;
	double** ttable3;
	cudaMalloc(&ttable1, 10 * sizeof(double*));
	cudaMalloc(&ttable2, 10 * sizeof(double*));
	cudaMalloc(&ttable3, 10 * sizeof(double*));
	cudaMemcpy(ttable1, &table1[D], 10 * sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(ttable2, &table2[D], 10 * sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(ttable3, &table3[D], 10 * sizeof(double*), cudaMemcpyHostToDevice);

	vec_ptr = thrust::raw_pointer_cast(d_nodes_d[D].data());
	compute_divergence << <numBlocks, threadsPerBlock >> > (D, vec_ptr, d_offset, ttable1, ttable2, ttable3, h_size_d[D]);
	cudaDeviceSynchronize();
	END_T();

	//vec_ptr = thrust::raw_pointer_cast(d_nodes_d[2].data());
	//see << <numBlocks, threadsPerBlock >> > (vec_ptr, h_size_d[2]);
	//cudaDeviceSynchronize();

	START_T("求解泊松方程");
	int pre_size[10] = { 0 };
	for (int d = 1; d <= D; d++) pre_size[d] = pre_size[d - 1] + h_size_d[d - 1];

	float* d_b;
	int* d_csrOffsets_before;
	int* d_csrOffsets;
	int* d_columns;
	float* d_values;
	cudaMalloc(&d_b, h_size_d[D] * sizeof(float));
	cudaMalloc(&d_csrOffsets_before, (h_size_d[D] + 1) * sizeof(int));
	cudaMemset(d_csrOffsets_before, 0, sizeof(int)); // 第一个元素赋值为0
	cudaMalloc(&d_csrOffsets, (h_size_d[D] + 1) * sizeof(int));
	cudaMalloc(&d_columns, 27 * h_size_d[D] * sizeof(int));
	cudaMalloc(&d_values, 27 * h_size_d[D] * sizeof(float));
	float* d_x[10];
	for (int d = 1; d <= D; d++) {
		cudaMalloc(&d_x[d], h_size_d[d] * sizeof(float));
	}
	for (int d = 1; d <= D; d++) {
		vec_ptr = thrust::raw_pointer_cast(d_nodes_d[d].data());
		set_b << <numBlocks, threadsPerBlock >> > (vec_ptr, d_b, h_size_d[d]);
		cudaDeviceSynchronize();
		for (int d2 = 1; d2 < d; d2++) {
			prepare_CSR << <numBlocks, threadsPerBlock >> > (vec_ptr, d, d2, d_csrOffsets_before, h_size_d[d]);
			thrust::device_ptr<int> d_ptr4(d_csrOffsets_before);
			thrust::device_ptr<int> d_ptr5(d_csrOffsets);
			thrust::inclusive_scan(thrust::device, d_ptr4, d_ptr4 + h_size_d[d] + 1, d_ptr5);
			construct_laplacian_matrix << <numBlocks, threadsPerBlock >> > (vec_ptr, d, d2, pre_size[d], pre_size[d2], d_offset, table1[d][d2], table2[d][d2], table3[d][d2], d_csrOffsets, d_columns, d_values, h_size_d[d]);
			cudaDeviceSynchronize();
			int num_rows = h_size_d[d];
			int num_cols = h_size_d[d2];
			int nnz;
			cudaMemcpy(&nnz, d_csrOffsets + h_size_d[d], sizeof(int), cudaMemcpyDeviceToHost);
			float*& dX = d_x[d2];
			float*& dY = d_b;
			SparseMatrixVectorMultiply(
				num_rows, num_cols, nnz,
				d_csrOffsets, d_columns, d_values,
				dX, dY, -1.0f, 1.0f);
		}
		prepare_CSR << <numBlocks, threadsPerBlock >> > (vec_ptr, d, d, d_csrOffsets_before, h_size_d[d]);
		thrust::device_ptr<int> d_ptr4(d_csrOffsets_before);
		thrust::device_ptr<int> d_ptr5(d_csrOffsets);
		thrust::inclusive_scan(thrust::device, d_ptr4, d_ptr4 + h_size_d[d] + 1, d_ptr5);
		construct_laplacian_matrix << <numBlocks, threadsPerBlock >> > (vec_ptr, d, d, pre_size[d], pre_size[d], d_offset, table1[d][d], table2[d][d], table3[d][d], d_csrOffsets, d_columns, d_values, h_size_d[d]);
		cudaDeviceSynchronize();
		solve_conjugate_gradient(d_csrOffsets, d_columns, d_values, d_b, d_x[d], h_size_d[d], int(pow(h_size_d[d], 1.0 / 3.0)));
		/*float* h_b = (float*)malloc(h_size_d[d] * sizeof(float));
		cudaMemcpy(h_b, d_b, h_size_d[d] * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < h_size_d[d]; i++) {
			std::cout << h_b[i] << std::endl;
		}
		std::cout << "-------------------" << std::endl;
		float* h_x = (float*)malloc(h_size_d[d] * sizeof(float));
		cudaMemcpy(h_x, d_x[d], h_size_d[d] * sizeof(float), cudaMemcpyDeviceToHost);
		for (int i = 0; i < h_size_d[d]; i++) {
			std::cout << h_x[i] << std::endl;
		}
		std::cout << std::endl;
		int xx;
		std::cin >> xx;*/
	}
	cudaFree(d_b);
	cudaFree(d_csrOffsets_before);
	cudaFree(d_csrOffsets);
	cudaFree(d_columns);
	cudaFree(d_values);
	// 计算solution
	float* d_solution;
	cudaMalloc(&d_solution, tot_size * sizeof(float));
	cudaMemset(d_solution, 0, sizeof(float)); // 第一个元素赋值为0
	for (int d = 1; d <= D; d++) {
		cudaMemcpy(d_solution + pre_size[d], d_x[d], h_size_d[d] * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	for (int d = 1; d <= D; d++) {
		cudaFree(d_x[d]);
	}
	END_T();

	START_T("构造函数值表");
	double* table4[10];
	int num = (1 << (D + 1)) + 1;
	double finest_width = 1.0 / (1 << (D + 1));
	for (int d = 0; d <= D; d++) {
		int len = (1 << d);
		cudaMalloc(&table4[d], len * num * sizeof(double));
		int size = len * num;
		int threadsPerBlock = 1024;
		int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
		set_value_table << <numBlocks, threadsPerBlock >> > (key_to_Fo[d], d, table4[d], finest_width, num, size);
		cudaDeviceSynchronize();
	}
	END_T();

	START_T("计算等值面的值");
	double** ttable4;
	cudaMalloc(&ttable4, 10 * sizeof(double*));
	cudaMemcpy(ttable4, table4, 10 * sizeof(double*), cudaMemcpyHostToDevice);

	double* d_sum;
	double* d_val;
	cudaMalloc(&d_sum, sizeof(double));
	cudaMalloc(&d_val, sizeof(double));
	cudaMemset(d_sum, 0, sizeof(double));
	cudaMemset(d_val, 0, sizeof(double));
	vec_ptr = thrust::raw_pointer_cast(d_nodes_d[D].data());
	thrust::host_vector<node*> h_vec = d_nodes_d[0];
	node* root = h_vec[0];
	compute_iso_value << <numBlocks, threadsPerBlock >> > (vec_ptr, d_sum, d_val, D, root, ttable4, d_offset, d_solution, h_size_d[D]);
	cudaDeviceSynchronize();
	double* h_sum = (double*)malloc(sizeof(double));
	double* h_val = (double*)malloc(sizeof(double));
	cudaMemcpy(h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_val, d_val, sizeof(double), cudaMemcpyDeviceToHost);
	double iso_value = (*h_val) / (*h_sum);
	std::cout << "等值面的值为" << iso_value << std::endl;
	END_T();

	START_T("为节点细分分配内存");
	int M = 256 * 256 * 256; // 可能被用到的节点索引的最大值，需要取上界256^3，其实不会全用到，但因为可能索引很大的节点被用到，所以还是得取满上界。除非可以用hash表优化？
	int M2 = 256 * 256 * 256; // 实际搜索到的节点数量的估计值，上界也是256^3，但它用作连续段的长度，不需要取满上界，可以乘估计系数0.01
	width = 1.0 / (1 << D);
	int* vis;
	int* has_root;
	cudaMalloc(&vis, M * sizeof(int));
	cudaMalloc(&has_root, M * sizeof(int));
	point* arr[2]; // 滚动数组，并行处理一个数组内的一批元素，同时把生成的一批元素放入另一个数组
	point* res;
	cudaMalloc(&arr[0], M2 * sizeof(point));
	cudaMalloc(&arr[1], M2 * sizeof(point));
	cudaMalloc(&res, M2 * sizeof(point));
	double* vals;
	double* res_vals;
	cudaMalloc(&vals, 8 * M * sizeof(double));
	cudaMalloc(&res_vals, 8 * M2 * sizeof(double));
	int* d_sz[2];
	int* d_res_sz;
	cudaMalloc(&d_sz[0], sizeof(int));
	cudaMalloc(&d_sz[1], sizeof(int));
	cudaMalloc(&d_res_sz, sizeof(int));
	cudaMemcpy(d_sz[0], &h_size_d[D], sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(d_res_sz, 0, sizeof(int));
	vec_ptr = thrust::raw_pointer_cast(d_nodes_d[D].data());
	put << <numBlocks, threadsPerBlock >> > (vec_ptr, arr[0], vis, width, D, h_size_d[D]);
	cudaDeviceSynchronize();
	END_T();
	START_T("节点细分");
	int turn = 0;
	int cur_sz = 0;
	while (1) {
		cudaMemcpy(&cur_sz, d_sz[turn], sizeof(int), cudaMemcpyDeviceToHost);
		if (cur_sz == 0) break;
		cudaMemset(d_sz[turn ^ 1], 0, sizeof(int)); // 保存生成结果的数组的初始有效长度为零
		int threadsPerBlock = 1024;
		int numBlocks = (8 * cur_sz + threadsPerBlock - 1) / threadsPerBlock;
		evaluate_vertices << <numBlocks, threadsPerBlock >> > (D, arr[turn], vals, d_sz[turn], width, root, ttable4, d_offset, d_solution, iso_value);
		cudaDeviceSynchronize();
		threadsPerBlock = 1024;
		numBlocks = (12 * cur_sz + threadsPerBlock - 1) / threadsPerBlock;
		extend << <numBlocks, threadsPerBlock >> > (D, vals, arr[turn], arr[turn ^ 1], d_sz[turn], d_sz[turn ^ 1], vis, has_root, width);
		cudaDeviceSynchronize();
		std::cout << cur_sz << std::endl;
		threadsPerBlock = 1024;
		numBlocks = (cur_sz + threadsPerBlock - 1) / threadsPerBlock;
		collect_useful_nodes << <numBlocks, threadsPerBlock >> > (arr[turn], vals, res, res_vals, d_sz[turn], d_res_sz, has_root, D, width);
		cudaDeviceSynchronize();
		turn ^= 1;
	}
	cudaMemcpy(&cur_sz, d_res_sz, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << cur_sz << std::endl;
	END_T();
	START_T("提取等值面");
	int* d_triangles_sz;
	cudaMalloc(&d_triangles_sz, sizeof(int));
	cudaMemset(d_triangles_sz, 0, sizeof(int));
	triangle* d_triangles;
	triangle* h_triangles;
	cudaMalloc(&d_triangles, 5 * cur_sz * sizeof(triangle));
	h_triangles = (triangle*)malloc(5 * cur_sz * sizeof(triangle));
	marching_cube << <numBlocks, threadsPerBlock >> > (width, res, res_vals, d_res_sz, d_triangles, d_triangles_sz);
	cudaDeviceSynchronize();
	cudaMemcpy(h_triangles, d_triangles, 5 * cur_sz * sizeof(triangle), cudaMemcpyDeviceToHost);
	cudaMemcpy(&cur_sz, d_triangles_sz, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << cur_sz << std::endl;
	END_T();

	// 为了方便，内存里保存的是一个个含有实际坐标点的三角形，而并没有按OBJ那种格式来保存唯一的顶点数组，然后保存三个顶点的索引
	// 因此保存为OBJ文件的过程很慢，这一点可以从一开始就更改内存中的保存形式来优化
	std::ofstream file;
	file.open("triangles.obj");
	struct point_comparator {
		bool operator()(const point& p1, const point& p2) const {
			const double threshold = 1e-8;
			if (fabs(p1.x - p2.x) >= threshold) {
				return p1.x < p2.x;
			}
			if (fabs(p1.y - p2.y) >= threshold) {
				return p1.y < p2.y;
			}
			if (fabs(p1.z - p2.z) >= threshold) {
				return p1.z < p2.z;
			}
			return false;
		}
	};
	std::map<point, int, point_comparator> triangle_vertices_idx;
	point* triangle_vertices;
	for (int j = 0; j < cur_sz; j++) {
		auto& triangle = h_triangles[j];
		for (int i = 0; i < 3; i++) {
			const point& p = triangle.p[i];
			if (triangle_vertices_idx.find(p) == triangle_vertices_idx.end()) {
				int idx = triangle_vertices_idx.size();
				triangle_vertices_idx[p] = idx;
			}
		}
	}
	triangle_vertices = new point[triangle_vertices_idx.size()]();
	for (const auto& pair : triangle_vertices_idx) {
		triangle_vertices[pair.second] = pair.first;
	}
	file << triangle_vertices_idx.size() << std::endl;
	for (int i = 0; i < triangle_vertices_idx.size(); i++) {
		point& p = triangle_vertices[i];
		file << "v " << p.x << " " << p.y << " " << p.z << std::endl;
	}
	for (int j = 0; j < cur_sz; j++) {
		auto& triangle = h_triangles[j];
		file << "f ";
		for (int i = 0; i < 3; i++) {
			const point& p = triangle.p[i];
			file << triangle_vertices_idx[triangle.p[i]] + 1 << " ";
		}
		file << std::endl;
	}
	return 0;
}
