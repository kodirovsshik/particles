
#define CL_TARGET_OPENCL_VERSION 110
#define CL_MINIMUM_OPENCL_VERSION 110



#include <ksn/window.hpp>
#include <ksn/opencl_selector.hpp>
#include <ksn/stuff.hpp>
#include <ksn/math_constants.hpp>
#include <ksn/try_smol_buffer.hpp>

#include <stdarg.h>
#include <string.h>

#include <GL/glew.h>

#include <chrono>

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
} particle_t;

__kernel void interaction(__global particle_t* p, float dt)
{
	__global particle_t* p1 = p + get_global_id(0);
	__global particle_t* p2 = p + get_global_id(1);

	if (p2 <= p1) return;

	float dx = p2->x - p1->x;
	float dy = p2->y - p1->y;
	float r = dx * dx + dy * dy;
	r = 1 / r;

	//float k = 6.6743e-11f * r;
	float a1 = r * p2->mass;
	float a2 = r * p1->mass;

	r = sqrt(r);
	
	float x_proj = dx * r;
	float y_proj = dy * r;
	
	p1->ax += x_proj * a1;
	p1->ay += y_proj * a1;

	p2->ax -= x_proj * a2;
	p2->ay -= y_proj * a2;
}

__kernel void movement(__global particle_t* arr, float dt)
{
	__global particle_t* p = arr + get_global_id(0);

	//dt *= 6.6743e-11f;

	p->vx += p->ax * 6.6743e-11f * dt;
	p->vy += p->ay * 6.6743e-11f * dt;
	
	p->x += p->vx * dt;
	p->y += p->vy * dt;

	p->ax = 0;
	p->ay = 0;
}

__kernel void collision_resolution(__global particle_t* p)
{
	size_t n1 = get_global_id(0);
	size_t n2 = get_global_id(1);

	if (n1 <= n2) return;

	__global particle_t* p1 = p + n1;
	__global particle_t* p2 = p + n2;

	float k1;
	float k2;

	float dx = p2->x - p1->x;
	float dy = p2->y - p1->y;

	{
		float r = sqrt(dx * dx + dy * dy);
		{
			float sr = p1->radius + p2->radius;
			if (r >= sr) return;
			r = (sr - r) / r;
		}

		float M1;
		float M2;

		{
			float v1 = sqrt(p1->vx * p1->vx + p1->vy * p1->vy);
			M1 = p1->mass * v1;
		}
		{
			float v2 = sqrt(p2->vx * p2->vx + p2->vy * p2->vy);
			M2 = p2->mass * v2;
		}
		if (M1 == 0 && M2 == 0)
		{
			M1 = M2 = 0.5;
		}
		else
		{
			float M = M1 + M2;
			if (M != 0)
			{
				M1 /= M;
				M2 /= M;
			}
		}
		
		//r *= 1.2;
		k1 = r * M1;
		k2 = r * M2;
	}

	p1->x -= dx * k1;
	p1->y -= dy * k1;

	p2->x += dx * k2;
	p2->y += dy * k2;

	p1->vx = 0;
	p1->vy = 0;
	p2->vx = 0;
	p2->vy = 0;
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

	size_t n_points = size_t(r_x > r_y ? r_x : r_y);
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
	do
	{
		char buffer_version[80];
		char buffer_device[80];
		char buffer_vendor[80];
		
		tempi = 0;
		clGetCommandQueueInfo(cl_q, CL_QUEUE_DEVICE, sizeof(cl_device), &cl_device, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_VERSION, 79, &buffer_version, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_NAME, 79, &buffer_device, nullptr);
		tempi |= clGetDeviceInfo(cl_device, CL_DEVICE_VENDOR, 79, &buffer_vendor, nullptr);
		
		if (tempi == 0) printf("OpenCL:\nRunning %s\n%s\n%s\n\n", buffer_version, buffer_device, buffer_vendor);
	} while (false);



	bool closed_view = false;

	std::vector<particle_t> particles;
	particles.reserve(64);

	if (false) //n circular layers of particles
	for (int i = 0, n = 5; i < n; ++i)
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
	{ //Rotation (ellipse-shaped)
		particle_t center{};
		center.x = width / 2;
		center.y = height / 2;
		center.mass = 1.5e16;
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
	if (true)
	{//Right triangle
		float r = 200;
		
		float a = KSN_PIf / 2;
		float da = KSN_PIf / 1.5f;

		float xc = width / 2, yc = height / 2;

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



	auto kernel_interact = clCreateKernel(cl_program, "interaction", nullptr);
	auto kernel_movement = clCreateKernel(cl_program, "movement", nullptr);
	auto kernel_collision_resolution = clCreateKernel(cl_program, "collision_resolution", nullptr);

	auto particles_buffer = clCreateBuffer(cl_context, 0, particles.size() * sizeof(particle_t), nullptr, nullptr);



	gluOrtho2D(0, width, height, 0);
	glClearColor(0.1f, 0.1f, 0.1f, 0);



	win.show();
	win.set_vsync_auto_or_enabled(true);


	std::vector<bool> key_pressed((int)ksn::keyboard_button_t::buttons_count, false);

	float zoom = 1;
	float dzoom = 1.1f;

	bool play = false;

	float dt_fps = 1.f / 60;

	size_t precision = 100;

	auto clock_f = &std::chrono::system_clock::now;
	auto t = clock_f();

	//TODO: make gravity work well
	do
	{
		auto t1 = clock_f();
		float dt = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t).count() / 1e9f;
		t = t1;

		if (particles.size() && play)
		{
			float dt_prec = dt_fps / precision;

			clEnqueueWriteBuffer(cl_q, particles_buffer, CL_FALSE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);

			tempi = clSetKernelArg(kernel_interact, 0, sizeof(particles_buffer), &particles_buffer);
			tempi = clSetKernelArg(kernel_interact, 1, sizeof(dt_prec), &dt_prec);

			tempi = clSetKernelArg(kernel_movement, 0, sizeof(particles_buffer), &particles_buffer);
			tempi = clSetKernelArg(kernel_movement, 1, sizeof(dt_prec), &dt_prec);

			tempi = clSetKernelArg(kernel_collision_resolution, 0, sizeof(particles_buffer), &particles_buffer);

			size_t global_work_offset[] = { 0, 0 };
			size_t local_work_size[] = { 1, 1 };
			size_t global_work_size[] = { particles.size(), particles.size() };

			for (size_t i = 0; i < precision; ++i)
			{
				tempi = clEnqueueNDRangeKernel(cl_q, kernel_interact, 2, global_work_offset, global_work_size, local_work_size, 0, nullptr, nullptr);
				tempi = clEnqueueNDRangeKernel(cl_q, kernel_movement, 1, global_work_offset, global_work_size, local_work_size, 0, nullptr, nullptr);
				tempi = clEnqueueNDRangeKernel(cl_q, kernel_collision_resolution, 2, global_work_offset, global_work_size, local_work_size, 0, nullptr, nullptr);
				tempi = clEnqueueNDRangeKernel(cl_q, kernel_collision_resolution, 2, global_work_offset, global_work_size, local_work_size, 0, nullptr, nullptr);
			}
			tempi = clEnqueueReadBuffer(cl_q, particles_buffer, CL_TRUE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);
			
			clFlush(cl_q);
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