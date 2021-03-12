
#define CL_TARGET_OPENCL_VERSION 110
#define CL_MINIMUM_OPENCL_VERSION 110

#include <ksn/window.hpp>
#include <ksn/opencl_selector.hpp>
#include <ksn/stuff.hpp>
#include <ksn/math_constants.hpp>


#include <stdarg.h>
#include <string.h>


#include <GL/glew.h>



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
	__global particle_t* p1 = p + get_global_size(0);
	__global particle_t* p2 = p1 + 1;

	float dx = p2->x - p1->x;
	float dy = p2->y - p1->y;
	float r = dx * dx + dy * dy;

	float k = 6.6743e-11f / r;
	float a1 = k * p2->mass;
	float a2 = k * p1->mass;

	r = 1 / sqrt(r);
	
	float x_proj = dx * r;
	float y_proj = dy * r;
	
	p1->ax += x_proj * a1;
	p1->ay += x_proj * a1;

	p2->ax += x_proj * a2;
	p2->ay += x_proj * a2;
}

)"
};

size_t cl_sources_lengthes[ksn::countof(cl_sources)];


size_t width = 800;
size_t height = 600;

float view_x = 0, view_y = 0, view_width_ratio = 1, view_height_ratio = 1;

void draw_circle(float x, float y, float r)
{
	x = (x - view_x) * view_width_ratio;
	y = (y - view_y) * view_height_ratio;

	float r_x = r * view_width_ratio;
	float r_y = r * view_height_ratio;

	size_t n_points = size_t(r_x > r_y ? r_x : r_y);
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

void set_view(float x1, float y1, float x2, float y2)
{
	view_x = x1;
	view_y = y1;
	view_width_ratio = width / (x2 - x1);
	view_height_ratio = height / (y2 - y1);
}


int main()
{

	ksn::window_t win;
	if (win.open(width, height, "", {},
		ksn::window_t::style::close_min_max | ksn::window_t::style::resize| ksn::window_t::style::hidden
	) != ksn::window_t::error::ok)
		ksn::error_exit(1, "Failed to open the window\n");

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
		ksn::error_exit(2, "Failed to find situable OpenCL implementation");


	auto cl_context = opencl_selector_data.context;
	auto cl_program = opencl_selector_data.program;
	auto cl_q = opencl_selector_data.q;


	win.make_current();
	win.show();
	win.set_vsync_auto_or_enabled(true);

	
	gluOrtho2D(0, width, height, 0);


	bool closed_view = false;

	std::vector<particle_t> particles;
	particles.reserve(64);
	
	for (int i = 0; i < 5; ++i)
	{
		const float r = 10;
		const float R = 25;

		const float center_x = width / 2;
		const float center_y = height / 2;

		const int n_spheres = i * 6 + (i == 0);
		float a = 0;
		const float da = 2 * KSN_PIf / n_spheres;

		for (int j = 0; j < n_spheres; ++j)
		{
			particle_t particle{};
			particle.x = center_x + i * R * cos(a);
			particle.y = center_y + i * R * sin(a);
			particle.radius = r;
			particles.push_back(particle);
			a += da;
		}
	}
	

	auto kernel_interact = clCreateKernel(cl_program, "interaction", nullptr);
	auto particles_buffer = clCreateBuffer(cl_context, 0, particles.size() * sizeof(particle_t), nullptr, nullptr);

	int tempi = 0;


	glClearColor(0, 0, 1, 0);

	while (win.is_open())
	{
		if (particles.size())
		{
			clEnqueueWriteBuffer(cl_q, particles_buffer, CL_FALSE, 0, particles.size() * sizeof(particle_t), particles.data(), 0, nullptr, nullptr);

			tempi = clSetKernelArg(kernel_interact, 0, sizeof(particles_buffer), &particles_buffer);
			float dt = 1.f / 60;
			tempi = clSetKernelArg(kernel_interact, 1, sizeof(dt), &dt);

			size_t zero = 0;
			size_t one = 1;
			size_t n1 = particles.size() - 1;

			tempi = clEnqueueNDRangeKernel(cl_q, kernel_interact, 1, &zero, &n1, &one, 0, nullptr, nullptr);
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



		//processing events
		ksn::event_t ev;
		while (win.poll_event(ev))
		{
			if (ev.type == ksn::event_type_t::keyboard_press)
			{
				if (ev.keyboard_button_data.button == ksn::event_t::keyboard_button_t::esc)	win.close();
				else if (ev.keyboard_button_data.button == ksn::event_t::keyboard_button_t::space)
				{
					if (closed_view)
						set_view(0, 0, width, height);
					else
						set_view(0, 0, 200, 200);
					closed_view ^= 1;
				}
			}
			else if (ev.type == ksn::event_type_t::close) win.close();
		}
	}

	return 0;
}