
#define CL_TARGET_OPENCL_VERSION 110
#define CL_MINIMUM_OPENCL_VERSION 110



#include <ksn/window.hpp>
#include <ksn/opencl_selector.hpp>
#include <ksn/stuff.hpp>
#include <ksn/math_constants.hpp>
#include <ksn/try_smol_buffer.hpp>
#include <ksn/math_common.hpp>

#include <stdarg.h>
#include <string.h>

#include <GL/glew.h>

#include <chrono>
#include <concepts>
#include <algorithm>
#include <random>

#include <Windows.h>


#pragma warning(disable : 26451)



_KSN_BEGIN

_KSN_DETAIL_BEGIN

[[noreturn]] void error_exit(int code, FILE* stream, const char* fmt, va_list ap)
{
	vfprintf(stream, fmt, ap);
	exit(code);
}

_KSN_DETAIL_END

[[noreturn]] void error_exit(int code, const char* fmt = "", ...)
{
	va_list ap;
	va_start(ap, fmt);
	detail::error_exit(code, stderr, fmt, ap);
}
[[noreturn]] void error_exit_stdout(int code, const char* fmt = "", ...)
{
	va_list ap;
	va_start(ap, fmt);
	detail::error_exit(code, stdout, fmt, ap);
}
[[noreturn]] void error_exit_file(int code, FILE* stream, const char* fmt = "", ...)
{
	va_list ap;
	va_start(ap, fmt);
	detail::error_exit(code, stream, fmt, ap);
}

_KSN_END



struct particle_t
{
	float x, y;
	float vx, vy;
	float ax, ay;
	float radius, mass;
#if _KSN_IS_DEBUG_BUILD
	float debug_float[8];
	uint32_t debug_uint[8];
#endif

	bool operator==(const particle_t& x) const noexcept
	{
		return memcmp(this, &x, sizeof(*this)) == 0;
	}
};

static_assert(sizeof(float) == sizeof(cl_float));



const char* const cl_sources[] =
{

	R"(
	
typedef struct
{
	float x, y;
	float vx, vy;
	float ax, ay;
	float radius, mass;
)"

#if _KSN_IS_DEBUG_BUILD
R"(
	float debug_float[8];
	unsigned int debug_uint[8];
)"
#endif
R"(
} particle_t;

//Computes how p[j] affects all other p[i]
__kernel void interaction(__global particle_t* p, float dt, unsigned int size, unsigned int j)
{
	unsigned int i = get_global_id(0);
	if (i >= size) return;

	//p[i].debug_uint[0] = i;
	//p[i].debug_uint[1] = j;
	//p[i].debug_uint[2] = get_global_id(0);
	//p[i].debug_uint[3] = get_global_size(0);
	//p[i].debug_uint[4] = get_local_size(0);
	//p[i].debug_uint[5] = get_local_id(0);
	//p[i].debug_uint[6] = size;

	if (i == j) return;

	__global particle_t* p1 = p + i;
	__global particle_t* p2 = p + j;

	float dx = p2->x - p1->x;
	float dy = p2->y - p1->y;
	float r_inv = (dx * dx + dy * dy);
	r_inv = 1 / r_inv;

	float a1 = r_inv * p2->mass;

	r_inv = sqrt(r_inv);
	
	float x_proj = dx * r_inv;
	float y_proj = dy * r_inv;
	
	p1->ax += x_proj * a1;
	p1->ay += y_proj * a1;
}

__kernel void movement(__global particle_t* arr, float dt, unsigned int size)
{
	unsigned int i = get_global_id(0);
	if (i >= size) return;

	__global particle_t* p = arr + i;
	//p->debug_float[0]

	float c = 6.6743e-11f * dt;

	float friction = 0.3f * (1 / 6.6743e-11f);
	p->ax -= p->vx * friction;
	p->ay -= p->vy * friction;

	p->vx += p->ax * c;
	p->vy += p->ay * c;

	p->x += p->vx * dt;
	p->y += p->vy * dt;

	p->ax = 0;
	p->ay = 0;
}

__kernel void collision_resolution(__global particle_t* p, unsigned int size)
{
	
}

__kernel void border(__global particle_t* p, unsigned int size, float x1, float x2, float y1, float y2)
{
	unsigned int i = get_global_id(0);
	if (i >= size) return;

	p += i;

	//p->debug_float[0] = x1;
	//p->debug_float[1] = x2;
	//p->debug_float[2] = y1;
	//p->debug_float[3] = y2;
	//p->debug_float[4] = p->x;
	//p->debug_float[5] = p->y;

	if (p->x < x1)
	{
		p->x = x1 + (x1 - p->x);
		if (p->vx < 0) p->vx *= -1;
	}
	if (p->x > x2)
	{
		p->x = x2 - (p->x - x2);
		if (p->vx > 0) p->vx *= -1;
	}
	if (p->y < y1)
	{
		p->y = y1 + (y1 - p->y);
		if (p->vy < 0) p->vy *= -1;
	}
	if (p->y > y2)
	{
		p->y = y2 - (p->y - y2);
		if (p->vy > 0) p->vy *= -1;
	}
}

)"

};

size_t cl_sources_lengthes[ksn::countof(cl_sources)];



uint16_t width = 800;
uint16_t height = 600;



struct
{
	float x = 0, y = 0, width_ratio = 1, height_ratio = 1;
} view_data;

void view_set_x(float x1, float x2)
{
	view_data.x = x1;
	view_data.width_ratio = width / (x2 - x1);
}
void view_set_y(float y1, float y2)
{
	view_data.y = y1;
	view_data.height_ratio = height / (y2 - y1);
}
void view_set(float x1, float x2, float y1, float y2)
{
	view_set_x(x1, x2);
	view_set_y(y1, y2);
}
void view_move_space(float dx, float dy)
{
	view_data.x += dx;
	view_data.y += dy;
}
void view_move_screen(float dx, float dy)
{
	view_data.x += dx / view_data.width_ratio;
	view_data.y += dy / view_data.height_ratio;
}
void view_zoom_relative(float z, float screen_x, float screen_y)
{
	view_move_screen(screen_x, screen_y);
	view_data.width_ratio *= z;
	view_data.height_ratio *= z;
	view_move_screen(-screen_x, -screen_y);
}
void view_zoom_center(float z)
{
	view_zoom_relative(z, (float)width / 2.f, (float)height / 2.f);
}




void draw_circle(float x, float y, float r)
{
	x = (x - view_data.x) * view_data.width_ratio;
	y = (y - view_data.y) * view_data.height_ratio;

	float r_x = r * view_data.width_ratio;
	float r_y = r * view_data.height_ratio;

	size_t n_points = size_t(r_x > r_y ? r_x : r_y) * 2;
	if (n_points < 8) n_points *= 3;
	float a = 0;
	const float da = 2 * KSN_PIf / n_points;
	
	glBegin(GL_POLYGON);

	for (size_t i = 0; i < n_points; ++i)
	{
		glVertex2f(x + r_x * cos(a), y + r_y * sin(a));
		a += da;
	}

	glEnd();
}




void GLAPIENTRY gl_debug_callback(GLenum, GLenum, GLuint code, GLenum, GLsizei, const GLchar* message, const void*)
{
	printf("OpenGL error %i:\n%s\n", code, message);
	__debugbreak();
}


_KSN_BEGIN

template<std::integral T>
constexpr T align_up(T value, T alignment)
{
	T remainder = value % alignment;
	if (remainder) value += alignment - remainder;
	return value;
	//Compiler, i hope for your CMOV support
}

_KSN_END


int main()
{
	printf("Loading\n");


	int tempi = 0;
	size_t tempsz = 0;


	ksn::window_t win;
	ksn::window_t::context_settings ogl;
	ksn::window_t::style_t window_style = ksn::window_t::style::close_min_max | ksn::window_t::style::resize | ksn::window_t::style::hidden;

	ogl.ogl_version_major = 4;
	ogl.ogl_version_minor = 3;
	ogl.ogl_debug = true;

	do
	{
		if (win.open(width, height, "", ogl, window_style) == ksn::window_t::error::ok)
			break;

		if (win.open(width, height, "", {}, window_style) == ksn::window_t::error::ok)
			break;
	} while (false);

	if (!win.is_open())
		ksn::error_exit(1, "Failed to open the window\n");

	win.make_current();


	glewExperimental = true;
	if (glewInit() != GLEW_OK)
		ksn::error_exit(2, "glewInit() has failed\n");



	if (glDebugMessageCallback)
		glDebugMessageCallback(gl_debug_callback, nullptr);
	

	for (size_t i = 0; i < ksn::countof(cl_sources); ++i)
	{
		cl_sources_lengthes[i] = strlen(cl_sources[i]);
	}


	ksn::opencl_selector_data_t opencl_selector_data;
	opencl_selector_data.opencl_major = 1;
	opencl_selector_data.opencl_minor = 1;
	opencl_selector_data.cl_build_parameters = "-cl-std=CL1.1";
	opencl_selector_data.cl_sources = cl_sources;
	opencl_selector_data.cl_sources_lengthes = cl_sources_lengthes;
	opencl_selector_data.build_log_file_name = L"OpenCL_build_log.txt";
	opencl_selector_data.cl_sources_number = 1;
	opencl_selector_data.platform = 1;
	opencl_selector_data.device = 1;
	if (ksn::opencl_selector(&opencl_selector_data) != 0)
		ksn::error_exit(3, "Failed to find situable OpenCL implementation");


	auto cl_context = opencl_selector_data.context;
	auto cl_program = opencl_selector_data.program;
	auto cl_q = opencl_selector_data.q;
	cl_device_id cl_device;


	printf("\nOpenGL:\nRunning OpenGL %s\n%s\n%s\n\n", glGetString(GL_VERSION), glGetString(GL_RENDERER), glGetString(GL_VENDOR));
	if constexpr (true)
	{
		char buffer_version[80];
		char buffer_device[80];
		char buffer_vendor[80];
		
		tempi = 0;
		clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_device), &cl_device, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_VERSION, sizeof(buffer_version), &buffer_version, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_NAME, sizeof(buffer_device), &buffer_device, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_VENDOR, sizeof(buffer_vendor), &buffer_vendor, nullptr);
		
		if (tempi == 0) printf("OpenCL:\nRunning %s\n%s\n%s\n\n", buffer_version, buffer_device, buffer_vendor);
	}



	bool closed_view = false;

	std::vector<particle_t> particles;
	particles.reserve(64);

	if (false) //n circular layers of particles
	//{} else
	for (int i = 0, n = 10; i < n; ++i)
	{ 
		const float r = 10;
		const float R = 25;

		const float center_x = width / 2.f;
		const float center_y = height / 2.f;

		const int n_spheres = i == 0 ? 1 : 6 * i;
		float a = 0;
		const float da = 2 * KSN_PIf / n_spheres;

		for (int j = 0; j < n_spheres; ++j)
		{
			particle_t particle{};
			particle.x = center_x + i * R * cos(a);
			particle.y = center_y + i * R * sin(a);
			particle.radius = r;
			particle.mass = 1e14f;

			particles.push_back(particle);
			a += da;
		}
	}
	if (false)
		//{} else
	{ //Just two particles
		float distance = 100;
		
		particle_t p{};
		p.x = width / 2.f - distance / 2;
		p.y = height/ 2.f;
		p.mass = 1e15f;
		p.radius = 10;
		particles.push_back(p);

		p.x += distance;
		particles.push_back(p);
	}
	if (false)
		//{} else
	{ //Rotation (ellipse-shaped)
		particle_t center{};
		center.x = width / 2.f;
		center.y = height / 2.f;
		center.mass = 1.5e16f;
		center.radius = 50;
		particles.push_back(center);

		particle_t obj{};
		obj.y = center.y - 100;
		obj.x = center.x;
		obj.vx = -100;
		obj.vy = 50;
		obj.radius = 5;
		particles.push_back(obj);
	}
	if (false)
		//{} else
	{//Right triangle
		float r = 200;
		
		float a = KSN_PIf / 2;
		float da = KSN_PIf / 1.5f;

		float xc = width / 2.f, yc = height / 2.f;

		for (int i = 0; i < 3; ++i)
		{
			particle_t vertex{};
			vertex.radius = 50;
			vertex.mass = 1.5e15f;
			vertex.x = xc + r * cos(a);
			vertex.y = yc - r * sin(a);
			
			particles.push_back(vertex);
			a += da;
		}
	}
	if (false)
		{ } else
	{
		constexpr static size_t N = 1000;
		
		std::random_device rd;
		std::mt19937_64 engine(rd());
		std::uniform_real_distribution<float> dist_x(0, width);
		std::uniform_real_distribution<float> dist_y(0, height);

		particle_t p{};
		p.radius = 5;
		p.mass = 1e13f;
		for (size_t i = 0; i < N; ++i)
		{
			p.x = dist_x(engine);
			p.y = dist_y(engine);
			particles.push_back(p);
		}
	}
	if (false)
	//{} else
	{
		particle_t p{};
		p.x = width / 2.f;
		p.y = height / 2.f;
		p.vx = 2000;
		p.radius = 10;
		p.mass = 1;

		particles.push_back(p);
	}

	const std::vector<particle_t> const_particles(particles);



	auto kernel_interact = clCreateKernel(cl_program, "interaction", nullptr);
	auto kernel_movement = clCreateKernel(cl_program, "movement", nullptr);
	auto kernel_collision_resolution = clCreateKernel(cl_program, "collision_resolution", nullptr);
	auto kernel_border = clCreateKernel(cl_program, "border", nullptr);

	auto particles_buffer = clCreateBuffer(cl_context, 0, particles.size() * sizeof(particle_t), nullptr, nullptr);



	gluOrtho2D(0, width, height, 0);
	glClearColor(0.1f, 0.1f, 0.1f, 0);



	win.show();
	//win.set_vsync_auto_or_enabled(true);


	std::vector<bool> key_pressed((int)ksn::keyboard_button_t::buttons_count, false);

	float zoom = 1;
	float dzoom = 1.1f;

	bool play = false;

	float dt_fps = 1.f / 60;

	size_t precision = 2;

	auto clock_f = &std::chrono::system_clock::now;
	auto t = clock_f();

	size_t max_work_group_size = 0;
	clGetDeviceInfo(cl_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
	const size_t local_work_size[] = { max_work_group_size};

	uint32_t particles_size = (uint32_t)particles.size();
	size_t global_work_size[] = { ksn::align_up<size_t>(particles_size, max_work_group_size) };

	tempi = clSetKernelArg(kernel_interact, 0, sizeof(particles_buffer), &particles_buffer);
	tempi = clSetKernelArg(kernel_interact, 2, sizeof(particles_size), &particles_size);

	tempi = clSetKernelArg(kernel_movement, 0, sizeof(particles_buffer), &particles_buffer);
	tempi = clSetKernelArg(kernel_movement, 2, sizeof(particles_size), &particles_size);

	tempi = clSetKernelArg(kernel_collision_resolution, 0, sizeof(particles_buffer), &particles_buffer);
	tempi = clSetKernelArg(kernel_collision_resolution, 1, sizeof(particles_size), &particles_size);

	tempi = clSetKernelArg(kernel_border, 0, sizeof(particles_buffer), &particles_buffer);
	tempi = clSetKernelArg(kernel_border, 1, sizeof(particles_size), &particles_size);
	{
		float temp;

		temp = 0;
		tempi = clSetKernelArg(kernel_border, 2, sizeof(temp), &temp);
		temp = width;
		tempi = clSetKernelArg(kernel_border, 3, sizeof(temp), &temp);
		temp = 0;
		tempi = clSetKernelArg(kernel_border, 4, sizeof(temp), &temp);
		temp = height;
		tempi = clSetKernelArg(kernel_border, 5, sizeof(temp), &temp);
	}

	clEnqueueWriteBuffer(cl_q, particles_buffer, CL_FALSE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);

	do
	{
		auto t1 = clock_f();
		float dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t).count() / 1e9f;
		t = t1;

		if (particles.size() && play)
		{
			float dt_prec = dt_fps / precision;

			tempi = clSetKernelArg(kernel_interact, 1, sizeof(dt_prec), &dt_prec);
			tempi = clSetKernelArg(kernel_movement, 1, sizeof(dt_prec), &dt_prec);

			size_t global_work_offset[2] = { 0, 0 };
			
			for (size_t j = 0; j < precision; ++j)
			{
				for (size_t i = 0; i < particles_size; ++i)
				{
					tempi = (int)i;
					tempi = clSetKernelArg(kernel_interact, 3, sizeof(tempi), &tempi);
					tempi = clEnqueueNDRangeKernel(cl_q, kernel_interact, 1, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
					//clFlush(cl_q);

					//if constexpr (
					//	false && 
					//	_KSN_IS_DEBUG_BUILD)
					//{
					//	tempi = clEnqueueReadBuffer(cl_q, particles_buffer, CL_TRUE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);
					//	auto* mismatch_elem = &*std::mismatch(particles.begin(), particles.end(), const_particles.begin()).first;
					//	size_t mismatch_index = mismatch_elem - particles.data();
					//	if (mismatch_index == particles_size) mismatch_index = -1;
					//	ksn::nop();
					//}
				}

				clEnqueueNDRangeKernel(cl_q, kernel_movement, 1, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
				//clFlush(cl_q);
				//_KSN_DEBUG_EXPR(tempi = clEnqueueReadBuffer(cl_q, particles_buffer, CL_TRUE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr));

				clEnqueueNDRangeKernel(cl_q, kernel_border, 1, global_work_offset, global_work_size, local_work_size, 0, NULL, NULL);
				//clFlush(cl_q);
				//_KSN_DEBUG_EXPR(tempi = clEnqueueReadBuffer(cl_q, particles_buffer, CL_TRUE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr));

				ksn::nop();
			}

			
			tempi = clEnqueueReadBuffer(cl_q, particles_buffer, CL_TRUE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);
			ksn::nop();

			printf("%i/10 ms\n", int(dt * 10000));
		}



		glClear(GL_COLOR_BUFFER_BIT);

		glColor3f(0, 1, 0);
		for (const auto &particle : particles)
		{
			draw_circle(particle.x, particle.y, particle.radius);
		}

		win.swap_buffers();



		ksn::event_t ev;
		while (win.poll_event(ev))
		{
			if (ev.type == ksn::event_type_t::keyboard_press)
			{
				switch (ev.keyboard_button_data.button)
				{
				case ksn::keyboard_button_t::esc:
					win.close();
					break;

				case ksn::keyboard_button_t::space:
					play = !play;
					break;
				}

				key_pressed[(int)ev.keyboard_button_data.button] = true;
			}
			if (ev.type == ksn::event_type_t::keyboard_release)
			{
				key_pressed[(int)ev.keyboard_button_data.button] = false;
			}
			else if (ev.type == ksn::event_type_t::close) win.close();
			else if (ev.type == ksn::event_type_t::resize)
			{
				width = ev.window_resize_data.width_new;
				height = ev.window_resize_data.height_new;

				glViewport(0, 0, width, height);
			}
			else if (ev.type == ksn::event_type_t::mouse_scroll_vertical)
			{
				float d = powf(dzoom, ev.mouse_scroll_data.delta);
				view_zoom_relative(d, ev.mouse_scroll_data.x, ev.mouse_scroll_data.y);
			}
		}

		float view_dx = 1000, view_dy = 1000;
		float view_delta_x = 0, view_delta_y = 0;
		if (key_pressed[(int)ksn::keyboard_button_t::arrow_left] || key_pressed[(int)ksn::keyboard_button_t::a])
			view_delta_x -= view_dx * dt_fps;
		if (key_pressed[(int)ksn::keyboard_button_t::arrow_right] || key_pressed[(int)ksn::keyboard_button_t::d])
			view_delta_x += view_dx * dt_fps;
		if (key_pressed[(int)ksn::keyboard_button_t::arrow_up] || key_pressed[(int)ksn::keyboard_button_t::w])
			view_delta_y -= view_dy * dt_fps;
		if (key_pressed[(int)ksn::keyboard_button_t::arrow_down] || key_pressed[(int)ksn::keyboard_button_t::s])
			view_delta_y += view_dy * dt_fps;
		view_move_screen(view_delta_x, view_delta_y);

	} 	while (win.is_open());
	
	

	return 0;
}



//OpenCL float atomics (fucking slow)

/*
// inline void atomic_add_gf2(volatile __global float *addr, float val)
//{
//	union
//	{
//		unsigned int u32;
//		float f32;
//	} next, expected, current;
//	current.f32 = *addr;
//	do
//	{
//		expected.f32 = current.f32;
//		next.f32 = expected.f32 + val;
//		current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr,
//		expected.u32, next.u32);
//	} while( current.u32 != expected.u32 );
//}
//
//inline float atomic_xchg_gf(__global float* addr, float val)
//{
//	int tmp = atomic_xchg((__global int*)addr, *(__private int*)&val);
//	return *(float*)&tmp;
//}
//
//static float atomic_cmpxchg_gf(volatile __global float *p, float cmp, float val) {
//	union {
//		unsigned int u32;
//		float        f32;
//	} cmp_union, val_union, old_union;
//
//	cmp_union.f32 = cmp;
//	val_union.f32 = val;
//	old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
//	return old_union.f32;
//}
//
//static float atomic_add_gf(volatile __global float *p, float val) {
//	float found = *p;
//	float expected;
//	do {
//		expected = found;
//		found = atomic_cmpxchg_gf(p, expected, expected + val);
//	} while (found != expected);
//	return found;
//}
//
// inline float atomic_add_gf1(volatile __global float *addr, float val)
//{
//	float data = *addr + val;
//	return atomic_xchg_gf(addr, data);
//}
*/
