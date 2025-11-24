import React, { useEffect, useRef, useState } from 'react';
import {
  ChevronLeft,
  Github,
  Linkedin,
  Mail,
  Download,
  Cpu,
  Wind,
  Activity,
  ChevronRight,
  Bookmark,
  Code,       // Added for the new section
  Terminal    // Added for the new section
} from 'lucide-react';

const App = () => {
  const canvasRef = useRef(null);
  const [view, setView] = useState('menu'); // menu, about, exp, projects, teaching, resume, meta
  const [isHovering, setIsHovering] = useState(false);

  // --- DATA CONTENT ---
  const EXPERIENCES = [
    {
      company: "Benchmark Space Systems",
      role: "Production Engineering Lead Intern",
      date: "2024",
      details: [
        "Architected a full-stack Manufacturing Execution System (MES) using React/PostgreSQL, replacing legacy processes and achieving 100% digital traceability for flight hardware.",
        "Engineered precision Ground Support Equipment (GSE) for 1.1N High-Performance Green Propulsion (HPGP) thruster lines.",
        "Reduced acceptance testing cycle time by 40% through the design of automated fluidic test fixtures."
      ]
    },
    {
      company: "UVM AIRLab",
      role: "Autonomy Research Fellow",
      date: "2024 - 2025",
      details: [
        "Developed non-linear state estimation algorithms (Adaptive Kalman Filters) for multi-rotor UAVs in GPS-denied environments.",
        "Executed outdoor flight validation campaigns, correlating sensor fusion telemetry with VICON ground truth data.",
        "Optimized C++ control loops for onboard embedded systems."
      ]
    },
    {
      company: "Linamar Corp",
      role: "Process Innovation Intern",
      date: "2023",
      details: [
        "Spearheaded the technical rollout of a Friction-Stir Welding line for automotive power units.",
        "Implemented Statistical Process Control (SPC) workflows that reduced defect rates across three production shifts."
      ]
    }
  ];

  const PROJECTS = [
    {
      title: "M.S. Thesis",
      sub: "Microgravity Fluid Dynamics",
      tech: "OpenFOAM / VOF Method",
      desc: "Developing novel CFD solvers to predict 'Fluid Hammer' events during propellant priming in deep-space propulsion systems. Validated against NASA experimental data."
    },
    {
      title: "Generative Thermal Design",
      sub: "Adjoint Optimization",
      tech: "ANSYS / Shape Opt.",
      desc: "Leveraging continuous adjoint solvers to biologically evolve internal cooling channels for high-flux inverters, minimizing pressure drop by 35%."
    },
    {
      title: "Propulsion OS",
      sub: "Full-Stack MES",
      tech: "React / Node / Postgres",
      desc: "A distributed hardware tracking system handling BOM management, non-conformance reporting, and acceptance test data for satellite thrusters."
    },
    {
      title: "Battery Thermal Runaway",
      sub: "Transient FEA",
      tech: "Thermal Simulation",
      desc: "Transient thermal analysis of 21700 cell pack propagation. Modeled phase change materials (PCM) to delay thermal runaway in high-density modules."
    }
  ];

  const TEACHING = [
    {
      title: "Undergraduate Materials Science Lab",
      sub: "Materials Science",
      desc: "I got my start teaching as a grad student in the undergraduate materials science lab. There I gave weekly lectures, ran experiments and ensured student safety. This served as my introduction to the world of teaching."
    },
    {
      title: "Reduced Order Modeling Course",
      sub: "Linear Analysis",
      desc: "As a personal project, I designed and released three new modules for the classic Lorena Barba 'Introduction to CFD Using Python course' covering the fundamentals of 'Data Driven Fluid Dynamics'."
    },
    {
      title: "Finite Element Analysis Course",
      sub: "Materials Science, FEA",
      desc: "I am the primary instructor for ME1510- Introduction to Finite Element Analysis at UVM. I give lectures, create assignments, and grade 80 students."
    },
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // --- CFD CONFIGURATION ---
    const N = 100;          // Grid Resolution (Higher = finer details, slower)
    const ITER = 15;        // Solver Iterations (Higher = more "incompressible" look)
    const SCALE = 4;        // Scale factor (controlled by canvas size)
    const VISCOSITY = 0.00001; // Fluid thickness
    const DIFFUSION = 0.0;  // Dye spreading (0 = stays sharp)
    const FADE_RATE = 0.015; // How fast the dye disappears
    const FORCE_SCALE = 5000; // Mouse push strength
    const DYE_AMOUNT = 1000;  // Amount of "smoke" added on interaction

    // --- SIMULATION STATE ---
    let size = (N + 2) * (N + 2);
    let u = new Float32Array(size);       // Velocity X
    let v = new Float32Array(size);       // Velocity Y
    let u_prev = new Float32Array(size);  // Previous Velocity X
    let v_prev = new Float32Array(size);  // Previous Velocity Y
    let dens = new Float32Array(size);    // Density (Dye)
    let dens_prev = new Float32Array(size); // Previous Density

    // --- SOLVER KERNELS (Stable Fluids Algorithm) ---
    // Converts 2D grid coordinates to 1D array index
    const IX = (x, y) => x + (N + 2) * y;

    // Handle Boundary Conditions (Reflection)
    const set_bnd = (b, x) => {
      for (let i = 1; i <= N; i++) {
        x[IX(0, i)] = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(N + 1, i)] = b === 1 ? -x[IX(N, i)] : x[IX(N, i)];
        x[IX(i, 0)] = b === 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N + 1)] = b === 2 ? -x[IX(i, N)] : x[IX(i, N)];
      }
      // Corners
      x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
      x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
      x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
      x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
    };

    // Linear Solver (Gauss-Seidel Relaxation)
    const lin_solve = (b, x, x0, a, c) => {
      const cRecip = 1.0 / c;
      for (let k = 0; k < ITER; k++) {
        for (let j = 1; j <= N; j++) {
          for (let i = 1; i <= N; i++) {
            x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] + x[IX(i, j - 1)])) * cRecip;
          }
        }
        set_bnd(b, x);
      }
    };

    // Diffusion Step
    const diffuse = (b, x, x0, diff, dt) => {
      const a = dt * diff * (N - 2) * (N - 2);
      lin_solve(b, x, x0, a, 1 + 6 * a);
    };

    // Projection Step (Enforces Mass Conservation / Incompressibility)
    const project = (u, v, p, div) => {
      const h = 1.0 / N;
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          div[IX(i, j)] = -0.5 * h * (u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
          p[IX(i, j)] = 0;
        }
      }
      set_bnd(0, div);
      set_bnd(0, p);
      lin_solve(0, p, div, 1, 4);

      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          u[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
          v[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
        }
      }
      set_bnd(1, u);
      set_bnd(2, v);
    };

    // Advection Step (Moving quantities along velocity field)
    const advect = (b, d, d0, u, v, dt) => {
      let i0, j0, i1, j1;
      let x, y, s0, t0, s1, t1;
      const dt0 = dt * (N - 2);
      const Nfloat = N;

      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          // Backtrace
          x = i - dt0 * u[IX(i, j)];
          y = j - dt0 * v[IX(i, j)];

          // Clamp
          if (x < 0.5) x = 0.5;
          if (x > Nfloat + 0.5) x = Nfloat + 0.5;
          i0 = Math.floor(x);
          i1 = i0 + 1;

          if (y < 0.5) y = 0.5;
          if (y > Nfloat + 0.5) y = Nfloat + 0.5;
          j0 = Math.floor(y);
          j1 = j0 + 1;

          // Bilinear Interpolation
          s1 = x - i0;
          s0 = 1.0 - s1;
          t1 = y - j0;
          t0 = 1.0 - t1;

          d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
            s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
      }
      set_bnd(b, d);
    };

    // --- INPUT HANDLING ---
    let prevMouseX = 0;
    let prevMouseY = 0;
    let hue = 0;

    const handleMouseMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      // Scale mouse coordinates to simulation grid
      const mx = Math.floor(((e.clientX - rect.left) / rect.width) * N) + 1;
      const my = Math.floor(((e.clientY - rect.top) / rect.height) * N) + 1;

      // Only act if within bounds
      if (mx > 0 && mx < N + 1 && my > 0 && my < N + 1) {
        // Calculate velocity based on mouse movement speed
        const dx = mx - prevMouseX;
        const dy = my - prevMouseY;

        // Add Sources
        const index = IX(mx, my);
        u[index] += dx * 2;
        v[index] += dy * 2;
        dens[index] += DYE_AMOUNT;

        // Add to neighbors for smoother appearance
        if (mx + 1 <= N) dens[IX(mx + 1, my)] += DYE_AMOUNT * 0.5;
        if (mx - 1 > 0) dens[IX(mx - 1, my)] += DYE_AMOUNT * 0.5;
        if (my + 1 <= N) dens[IX(mx, my + 1)] += DYE_AMOUNT * 0.5;
        if (my - 1 > 0) dens[IX(mx, my - 1)] += DYE_AMOUNT * 0.5;

        prevMouseX = mx;
        prevMouseY = my;
      }
    };


    // NEW: Handle Click for "Splash" effect
    const handleClick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = Math.floor(((e.clientX - rect.left) / rect.width) * N) + 1;
      const my = Math.floor(((e.clientY - rect.top) / rect.height) * N) + 1;

      // Parameters for the splash
      const splashRadius = 4;
      const splashForce = 15; // Velocity multiplier
      const splashDensity = 500; // Extra dye

      for (let i = -splashRadius; i <= splashRadius; i++) {
        for (let j = -splashRadius; j <= splashRadius; j++) {
          const cx = mx + i;
          const cy = my + j;

          if (cx > 0 && cx < N + 1 && cy > 0 && cy < N + 1) {
            const dist = Math.sqrt(i * i + j * j);
            if (dist <= splashRadius) {
              const idx = IX(cx, cy);
              // Add lots of dye
              dens[idx] += splashDensity;
              // Add radial explosive velocity
              // Direction is (i, j) normalized roughly
              u[idx] += i * splashForce;
              v[idx] += j * splashForce;
            }
          }
        }
      }
    };

    // --- MAIN LOOP ---
    const dt = 0.1;

    const step = () => {
      // Velocity Step
      diffuse(1, u_prev, u, VISCOSITY, dt);
      diffuse(2, v_prev, v, VISCOSITY, dt);
      project(u_prev, v_prev, u, v);
      advect(1, u, u_prev, u_prev, v_prev, dt);
      advect(2, v, v_prev, u_prev, v_prev, dt);
      project(u, v, u_prev, v_prev);

      // Density Step
      diffuse(0, dens_prev, dens, DIFFUSION, dt);
      advect(0, dens, dens_prev, u, v, dt);

      // Dissipation
      for (let i = 0; i < size; i++) {
        dens[i] *= (1 - FADE_RATE);
      }

      // Cycle Color
      hue = (hue + 0.5) % 360;
    };

    const render = () => {
      // Clear
      ctx.fillStyle = '#020617'; // Slate-950
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const cellW = Math.ceil(canvas.width / N);
      const cellH = Math.ceil(canvas.height / N);

      for (let i = 1; i <= N; i++) {
        for (let j = 1; j <= N; j++) {
          const d = dens[IX(i, j)];
          if (d > 0.1) {
            // Made alpha slightly more aggressive for better definition
            const alpha = Math.min(d / 200, 0.9);
            // Dynamic HSL Color - Darker Lightness (40% instead of 60%)
            ctx.fillStyle = `hsla(${hue}, 85%, 40%, ${alpha})`;
            // Overlap slightly to remove grid lines
            ctx.fillRect((i - 1) * cellW, (j - 1) * cellH, cellW, cellH);
          }
        }
      }
    };

    let animationId;
    const loop = () => {
      step();
      render();
      animationId = requestAnimationFrame(loop);
    };

    const handleResize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    // Initialization
    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mousedown', handleClick); // Added listener
    handleResize();
    loop();

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mousedown', handleClick); // Remove listener
      cancelAnimationFrame(animationId);
    };
  }, []);

  // --- SUBTLE UI COMPONENTS ---

  const SubtleCard = ({ children, className = "" }) => (
    <div className={`
      relative backdrop-blur-md bg-black/20 border border-white/5 rounded-sm
      transition-all duration-700 animate-in fade-in zoom-in-95
      ${className}
    `}>
      {children}
    </div>
  );

  const NavItem = ({ label, onClick, active }) => (
    <button
      onClick={onClick}
      className={`
        group relative flex items-center gap-4 py-4 px-6 w-full text-left
        transition-all duration-300 border-l-2
        ${active ? 'border-cyan-500 bg-white/5' : 'border-transparent hover:border-white/20 hover:bg-white/5'}
      `}
    >
      <span className={`font-mono uppercase tracking-[0.2em] text-sm ${active ? 'text-cyan-400' : 'text-slate-500 group-hover:text-slate-300'}`}>
        {label}
      </span>
      {active && <ChevronRight className="text-cyan-500 ml-auto animate-pulse" size={16} />}
    </button>
  );

  const BackLink = () => (
    <button
      onClick={() => setView('menu')}
      className="mb-12 flex items-center gap-3 text-slate-600 hover:text-cyan-400 transition-colors group"
    >
      <ChevronLeft size={14} className="group-hover:-translate-x-1 transition-transform" />
      <span className="font-mono text-xs uppercase tracking-widest">Back</span>
    </button>
  );

  // --- VIEW CONTENTS ---

  const renderContent = () => {
    if (view === 'menu') {
      return (
        <div className="flex flex-col items-start justify-center h-full max-w-5xl w-full mx-auto px-8 md:px-16">
          {/* HERO TEXT - NO CARD */}
          <div className="mb-24 mix-blend-overlay">
            <h1 className="text-7xl md:text-9xl font-bold text-white tracking-tighter opacity-80">
              GABE
            </h1>
            <h1 className="text-7xl md:text-9xl font-bold text-white/50 tracking-tighter -mt-4 md:-mt-8">
              JOHNSON
            </h1>
          </div>

          {/* NAVIGATION - MINIMAL LIST */}
          <div className="w-full max-w-sm border-t border-white/10 pt-4">
            <NavItem label="About Me" onClick={() => setView('about')} />
            <NavItem label="My Work" onClick={() => setView('experience')} />
            <NavItem label="Projects" onClick={() => setView('projects')} />
            <NavItem label="Teaching" onClick={() => setView('teaching')} />
            <NavItem label="Contact" onClick={() => setView('resume')} />

          </div>

          {/* SOCIALS - FLOATING */}
          <div className="fixed bottom-12 right-12 flex gap-8 text-slate-600">
            <a href="https://github.com/gabejohnsnn" className="hover:text-cyan-400 transition-colors"><Github size={20} /></a>
            <a href="https://substack.com/gabejohnsnn" className="hover:text-cyan-400 transition-colors"><Bookmark size={20} /></a>
            <a href="https://www.linkedin.com/in/gabojohnson/" className="hover:text-cyan-400 transition-colors"><Linkedin size={20} /></a>
            <a href="mailto:gabriel.johnson@uvm.edu" className="hover:text-cyan-400 transition-colors"><Mail size={20} /></a>
          </div>
        </div>
      );
    }

    // CONTENT VIEWS
    return (
      <div className="h-full w-full overflow-y-auto px-6 py-12 md:p-24">
        <div className="max-w-4xl mx-auto">
          <BackLink />

          {view === 'about' && (
            <SubtleCard className="p-8 md:p-16">
              <div className="flex items-center gap-4 mb-8 text-cyan-500/50">
                <Cpu size={24} />
                <span className="font-mono text-xs uppercase tracking-widest"></span>
              </div>
              <p className="text-2xl md:text-4xl font-light text-slate-200 leading-tight mb-12">
                Hi, I'm <span className="text-cyan-400 font-normal">Gabe</span>. I'm a creator, engineer, and <span className="text-cyan-400 font-normal">lifelong student</span>.
              </p>
              <div className="grid md:grid-cols-2 gap-12 font-mono text-sm text-slate-400 leading-relaxed">
                <p>
                  I am currently an M.S. candidate at UVM, specializing in data driven fluid dynamics for small
                  propulsion systems.  I am a lifelong polymath, and my work ranges from CFD to materials science,
                  philosophy to spirituality.

                  In my free time, you can find me listening to or playing bluegrass music, spening time in
                  the great outdoors, or reading. I have spent a lot of my life traveling, and have written about it
                  on my Substack.
                </p>
                <ul className="space-y-4 border-l border-white/10 pl-8">
                  <li className="block">
                    <span className="text-slate-600 block text-xs uppercase mb-1">Core Focus</span>
                    Fluid Dynamics & Thermal Systems
                  </li>
                  <li className="block">
                    <span className="text-slate-600 block text-xs uppercase mb-1">Tooling</span>
                    ANSYS, OpenFOAM, Python, React
                  </li>
                </ul>
              </div>
            </SubtleCard>
          )}

          {view === 'experience' && (
            <div className="space-y-4">
              {EXPERIENCES.map((job, i) => (
                <SubtleCard key={i} className="p-8 group hover:bg-white/5 transition-colors">
                  <div className="flex flex-col md:flex-row md:items-baseline justify-between mb-4">
                    <h3 className="text-xl font-medium text-slate-200 group-hover:text-cyan-400 transition-colors">{job.company}</h3>
                    <span className="font-mono text-xs text-slate-600">{job.date}</span>
                  </div>
                  <div className="mb-6 font-mono text-xs text-cyan-500/70 uppercase tracking-widest">{job.role}</div>
                  <div className="space-y-2">
                    {job.details.map((d, j) => (
                      <p key={j} className="text-slate-400 text-sm font-light">— {d}</p>
                    ))}
                  </div>
                </SubtleCard>
              ))}
            </div>
          )}

          {view === 'projects' && (
            <div className="grid md:grid-cols-2 gap-4">
              {PROJECTS.map((proj, i) => (
                <SubtleCard key={i} className="p-8 flex flex-col h-64 justify-between hover:border-cyan-500/30 group">
                  <div>
                    <div className="font-mono text-xs text-cyan-500 mb-2">{proj.sub}</div>
                    <h3 className="text-2xl text-slate-200 font-light mb-4 group-hover:text-white transition-colors">{proj.title}</h3>
                    <p className="text-sm text-slate-400 leading-relaxed">{proj.desc}</p>
                  </div>
                  <div className="font-mono text-xs text-slate-600 pt-4 border-t border-white/5 mt-4">
                    {proj.tech}
                  </div>
                </SubtleCard>
              ))}
            </div>
          )}

          {view === 'teaching' && (
            <div className="grid md:grid-cols-2 gap-4">
              {TEACHING.map((proj, i) => (
                <SubtleCard key={i} className="p-8 flex flex-col h-64 justify-between hover:border-cyan-500/30 group">
                  <div>
                    <div className="font-mono text-xs text-cyan-500 mb-2">{proj.sub}</div>
                    <h3 className="text-2xl text-slate-200 font-light mb-4 group-hover:text-white transition-colors">{proj.title}</h3>
                    <p className="text-sm text-slate-400 leading-relaxed">{proj.desc}</p>
                  </div>
                </SubtleCard>
              ))}
            </div>
          )}

          {view === 'resume' && (
            <div className="flex items-center justify-center h-[50vh]">
              <SubtleCard className="p-12 text-center hover:border-cyan-500/50 group cursor-pointer">
                <Download className="mx-auto mb-6 text-slate-600 group-hover:text-cyan-400 transition-colors" size={32} />
                <h3 className="text-xl text-slate-200 font-light mb-2">GJ_RESUME_2025.PDF</h3>
                <p className="font-mono text-xs text-slate-500 uppercase tracking-widest mb-8">Secure Document // 4.2 MB</p>
                <span className="px-6 py-3 bg-white/5 text-xs font-mono text-cyan-400 uppercase tracking-widest rounded-sm group-hover:bg-cyan-500/10 transition-colors">
                  Initiate Download
                </span>
              </SubtleCard>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500/30">
      {/* FLUID BACKGROUND */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* MAIN CONTENT OVERLAY */}
      <div className="relative z-10 w-full h-full">
        {renderContent()}
      </div>
    </div>
  );
};

export default App;