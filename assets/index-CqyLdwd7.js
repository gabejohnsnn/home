import { useRef, useState, useEffect } from "react";
import {
  Bookmark,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Download,
  Github,
  Linkedin,
  Mail,
} from "lucide-react";

// DATA

const EXPERIENCE = [
  {
    company: "Benchmark Space Systems",
    role: "Production Engineering Lead Intern",
    date: "2024",
    details: [
      "Architected a full-stack Manufacturing Execution System (MES) using React/PostgreSQL, replacing legacy processes and achieving 100% digital traceability for flight hardware.",
      "Engineered precision Ground Support Equipment (GSE) for 1.1N High-Performance Green Propulsion (HPGP) thruster lines.",
      "Reduced acceptance testing cycle time by 40% through the design of automated fluidic test fixtures.",
    ],
  },
  {
    company: "UVM AIRLab",
    role: "Autonomy Research Fellow",
    date: "2024 - 2025",
    details: [
      "Developed non-linear state estimation algorithms (Adaptive Kalman Filters) for multi-rotor UAVs in GPS-denied environments.",
      "Executed outdoor flight validation campaigns, correlating sensor fusion telemetry with VICON ground truth data.",
      "Optimized C++ control loops for onboard embedded systems.",
    ],
  },
  {
    company: "Linamar Corp",
    role: "Process Innovation Intern",
    date: "2023",
    details: [
      "Spearheaded the technical rollout of a Friction-Stir Welding line for automotive power units.",
      "Implemented Statistical Process Control (SPC) workflows that reduced defect rates across three production shifts.",
    ],
  },
];

const PROJECTS = [
  {
    title: "M.S. Thesis",
    sub: "Microgravity Fluid Dynamics",
    tech: "OpenFOAM / VOF Method",
    desc: "Developing novel CFD solvers to predict 'Fluid Hammer' events during propellant priming in deep-space propulsion systems. Validated against NASA experimental data.",
  },
  {
    title: "Generative Thermal Design",
    sub: "Adjoint Optimization",
    tech: "ANSYS / Shape Opt.",
    desc: "Leveraging continuous adjoint solvers to biologically evolve internal cooling channels for high-flux inverters, minimizing pressure drop by 35%.",
  },
  {
    title: "Propulsion OS",
    sub: "Full-Stack MES",
    tech: "React / Node / Postgres",
    desc: "A distributed hardware tracking system handling BOM management, non-conformance reporting, and acceptance test data for satellite thrusters.",
  },
  {
    title: "Battery Thermal Runaway",
    sub: "Transient FEA",
    tech: "Thermal Simulation",
    desc: "Transient thermal analysis of 21700 cell pack propagation. Modeled phase change materials (PCM) to delay thermal runaway in high-density modules.",
  },
];

const TEACHING = [
  {
    title: "Undergraduate Materials Science Lab",
    sub: "Materials Science",
    desc: "I got my start teaching as a grad student in the undergraduate materials science lab. There I gave weekly lectures, ran experiments and ensured student safety. This served as my introduction to the world of teaching.",
  },
  {
    title: "Reduced Order Modeling Course",
    sub: "Linear Analysis",
    desc: "As a personal project, I designed and released three new modules for the classic Lorena Barba 'Introduction to CFD Using Python' course covering the fundamentals of 'Data Driven Fluid Dynamics'.",
  },
  {
    title: "Finite Element Analysis Course",
    sub: "Materials Science, FEA",
    desc: "I am the primary instructor for ME1510 – Introduction to Finite Element Analysis at UVM. I give lectures, create assignments, and grade 80 students.",
  },
];

// FLUID SIMULATION (2D Navier-Stokes)

const GRID = 100;         // grid cells per axis
const ITERATIONS = 15;    // pressure solver iterations
const DIFFUSION = 1e-5;
const VISCOSITY = 0;
const DISSIPATION = 0.015;
const DYE_AMOUNT = 1000;

function useFluidCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const N = GRID;
    const size = (N + 2) * (N + 2);
    let velX     = new Float32Array(size);
    let velY     = new Float32Array(size);
    let velXPrev = new Float32Array(size);
    let velYPrev = new Float32Array(size);
    let dye      = new Float32Array(size);
    let dyePrev  = new Float32Array(size);

    let hue = 0;
    let prevMouseX = 0;
    let prevMouseY = 0;

    // flat index helper
    const idx = (x, y) => x + (N + 2) * y;

    // enforce boundary conditions
    function setBoundary(b, field) {
      for (let i = 1; i <= N; i++) {
        field[idx(0, i)]     = b === 1 ? -field[idx(1, i)]     : field[idx(1, i)];
        field[idx(N + 1, i)] = b === 1 ? -field[idx(N, i)]     : field[idx(N, i)];
        field[idx(i, 0)]     = b === 2 ? -field[idx(i, 1)]     : field[idx(i, 1)];
        field[idx(i, N + 1)] = b === 2 ? -field[idx(i, N)]     : field[idx(i, N)];
      }
      field[idx(0, 0)]         = 0.5 * (field[idx(1, 0)]         + field[idx(0, 1)]);
      field[idx(0, N + 1)]     = 0.5 * (field[idx(1, N + 1)]     + field[idx(0, N)]);
      field[idx(N + 1, 0)]     = 0.5 * (field[idx(N, 0)]         + field[idx(N + 1, 1)]);
      field[idx(N + 1, N + 1)] = 0.5 * (field[idx(N, N + 1)]     + field[idx(N + 1, N)]);
    }

    // Gauss-Seidel linear solver
    function linSolve(b, field, fieldPrev, a, c) {
      const inv = 1 / c;
      for (let iter = 0; iter < ITERATIONS; iter++) {
        for (let j = 1; j <= N; j++) {
          for (let i = 1; i <= N; i++) {
            field[idx(i, j)] =
              (fieldPrev[idx(i, j)] +
                a * (field[idx(i + 1, j)] + field[idx(i - 1, j)] +
                     field[idx(i, j + 1)] + field[idx(i, j - 1)])) * inv;
          }
        }
        setBoundary(b, field);
      }
    }

    function diffuse(b, field, fieldPrev, diff, dt) {
      const a = dt * diff * (N - 2) * (N - 2);
      linSolve(b, field, fieldPrev, a, 1 + 6 * a);
    }

    function project(vx, vy, vxPrev, vyPrev) {
      const h = 1 / N;
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          vyPrev[idx(i, j)] =
            -0.5 * h *
            (vx[idx(i + 1, j)] - vx[idx(i - 1, j)] +
             vy[idx(i, j + 1)] - vy[idx(i, j - 1)]);
          vxPrev[idx(i, j)] = 0;
        }
      }
      setBoundary(0, vyPrev);
      setBoundary(0, vxPrev);
      linSolve(0, vxPrev, vyPrev, 1, 4);

      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          vx[idx(i, j)] -= 0.5 * (vxPrev[idx(i + 1, j)] - vxPrev[idx(i - 1, j)]) / h;
          vy[idx(i, j)] -= 0.5 * (vxPrev[idx(i, j + 1)] - vxPrev[idx(i, j - 1)]) / h;
        }
      }
      setBoundary(1, vx);
      setBoundary(2, vy);
    }

    function advect(b, field, fieldPrev, vx, vy, dt) {
      const dtN = dt * (N - 2);
      const Nf = N;
      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          let x = i - dtN * vx[idx(i, j)];
          let y = j - dtN * vy[idx(i, j)];
          x = Math.max(0.5, Math.min(Nf + 0.5, x));
          y = Math.max(0.5, Math.min(Nf + 0.5, y));
          const i0 = Math.floor(x), i1 = i0 + 1;
          const j0 = Math.floor(y), j1 = j0 + 1;
          const sx = x - i0, tx = 1 - sx;
          const sy = y - j0, ty = 1 - sy;
          field[idx(i, j)] =
            tx * (ty * fieldPrev[idx(i0, j0)] + sy * fieldPrev[idx(i0, j1)]) +
            sx * (ty * fieldPrev[idx(i1, j0)] + sy * fieldPrev[idx(i1, j1)]);
        }
      }
      setBoundary(b, field);
    }

    const DT = 0.1;

    function stepSimulation() {
      // velocity
      diffuse(1, velXPrev, velX, VISCOSITY, DT);
      diffuse(2, velYPrev, velY, VISCOSITY, DT);
      project(velXPrev, velYPrev, velX, velY);
      advect(1, velX, velXPrev, velXPrev, velYPrev, DT);
      advect(2, velY, velYPrev, velXPrev, velYPrev, DT);
      project(velX, velY, velXPrev, velYPrev);

      // dye
      diffuse(0, dyePrev, dye, DIFFUSION, DT);
      advect(0, dye, dyePrev, velX, velY, DT);
      for (let i = 0; i < size; i++) dye[i] *= (1 - DISSIPATION);

      hue = (hue + 0.5) % 360;
    }

    function render() {
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const cellW = Math.ceil(canvas.width  / N);
      const cellH = Math.ceil(canvas.height / N);

      for (let j = 1; j <= N; j++) {
        for (let i = 1; i <= N; i++) {
          const d = dye[idx(i, j)];
          if (d > 0.1) {
            const alpha = Math.min(d / 200, 0.9);
            ctx.fillStyle = `hsla(${hue}, 85%, 40%, ${alpha})`;
            ctx.fillRect((i - 1) * cellW, (j - 1) * cellH, cellW, cellH);
          }
        }
      }
    }

    function handleMouseMove(e) {
      const rect = canvas.getBoundingClientRect();
      const cx = Math.floor(((e.clientX - rect.left) / rect.width)  * N) + 1;
      const cy = Math.floor(((e.clientY - rect.top)  / rect.height) * N) + 1;

      if (cx > 0 && cx < N + 1 && cy > 0 && cy < N + 1) {
        const dx = cx - prevMouseX;
        const dy = cy - prevMouseY;
        const i = idx(cx, cy);
        velX[i] += dx * 2;
        velY[i] += dy * 2;
        dye[i]  += DYE_AMOUNT;
        if (cx + 1 <= N)     dye[idx(cx + 1, cy)] += DYE_AMOUNT * 0.5;
        if (cx - 1 > 0)      dye[idx(cx - 1, cy)] += DYE_AMOUNT * 0.5;
        if (cy + 1 <= N)     dye[idx(cx, cy + 1)] += DYE_AMOUNT * 0.5;
        if (cy - 1 > 0)      dye[idx(cx, cy - 1)] += DYE_AMOUNT * 0.5;
        prevMouseX = cx;
        prevMouseY = cy;
      }
    }

    function handleMouseDown(e) {
      const rect  = canvas.getBoundingClientRect();
      const cx    = Math.floor(((e.clientX - rect.left) / rect.width)  * N) + 1;
      const cy    = Math.floor(((e.clientY - rect.top)  / rect.height) * N) + 1;
      const R     = 4;
      const speed = 15;
      const burst = 500;

      for (let dx = -R; dx <= R; dx++) {
        for (let dy = -R; dy <= R; dy++) {
          const nx = cx + dx, ny = cy + dy;
          if (nx > 0 && nx < N + 1 && ny > 0 && ny < N + 1 &&
              Math.sqrt(dx * dx + dy * dy) <= R) {
            const i = idx(nx, ny);
            dye[i]  += burst;
            velX[i] += dx * speed;
            velY[i] += dy * speed;
          }
        }
      }
    }

    function resize() {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    }

    let raf;
    function loop() {
      stepSimulation();
      render();
      raf = requestAnimationFrame(loop);
    }

    window.addEventListener("resize",    resize);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mousedown", handleMouseDown);
    resize();
    loop();

    return () => {
      window.removeEventListener("resize",    resize);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mousedown", handleMouseDown);
      cancelAnimationFrame(raf);
    };
  }, []);

  return canvasRef;
}

// SHARED UI COMPONENTS

function Card({ children, className = "" }) {
  return (
    <div
      className={`
        relative backdrop-blur-md bg-black/20 border border-white/5 rounded-sm
        transition-all duration-700 animate-in fade-in zoom-in-95
        ${className}
      `}
    >
      {children}
    </div>
  );
}

function NavItem({ label, onClick, active }) {
  return (
    <button
      onClick={onClick}
      className={`
        group relative flex items-center gap-4 py-4 px-6 w-full text-left
        transition-all duration-300 border-l-2
        ${active
          ? "border-cyan-500 bg-white/5"
          : "border-transparent hover:border-white/20 hover:bg-white/5"}
      `}
    >
      <span
        className={`font-mono uppercase tracking-[0.2em] text-sm ${
          active ? "text-cyan-400" : "text-slate-500 group-hover:text-slate-300"
        }`}
      >
        {label}
      </span>
      {active && (
        <ChevronRight className="text-cyan-500 ml-auto animate-pulse" size={16} />
      )}
    </button>
  );
}

function BackButton({ onClick }) {
  return (
    <button
      onClick={onClick}
      className="mb-12 flex items-center gap-3 text-slate-600 hover:text-cyan-400 transition-colors group"
    >
      <ChevronLeft size={14} className="group-hover:-translate-x-1 transition-transform" />
      <span className="font-mono text-xs uppercase tracking-widest">Back</span>
    </button>
  );
}

// SECTION VIEWS
function AboutView() {
  return (
    <Card className="p-8 md:p-16">
      <div className="flex items-center gap-4 mb-8 text-cyan-500/50">
        <Cpu size={24} />
        <span className="font-mono text-xs uppercase tracking-widest" />
      </div>

      <p className="text-2xl md:text-4xl font-light text-slate-200 leading-tight mb-12">
        Hi, I'm <span className="text-cyan-400 font-normal">Gabe</span>.
        I'm a creator, engineer, and{" "}
        <span className="text-cyan-400 font-normal">lifelong student</span>.
      </p>

      <div className="grid md:grid-cols-2 gap-12 font-mono text-sm text-slate-400 leading-relaxed">
        <p>
          I am currently an M.S. candidate at UVM, specializing in data driven
          fluid dynamics for small propulsion systems. I am a lifelong polymath,
          and my work ranges from CFD to materials science, philosophy to
          spirituality. In my free time, you can find me listening to or playing
          bluegrass music, spending time in the great outdoors, or reading. I
          have spent a lot of my life traveling, and have written about it on my
          Substack.
        </p>

        <ul className="space-y-4 border-l border-white/10 pl-8">
          <li className="block">
            <span className="text-slate-600 block text-xs uppercase mb-1">Core Focus</span>
            Fluid Dynamics &amp; Thermal Systems
          </li>
          <li className="block">
            <span className="text-slate-600 block text-xs uppercase mb-1">Tooling</span>
            ANSYS, OpenFOAM, Python, React
          </li>
        </ul>
      </div>
    </Card>
  );
}

function ExperienceView() {
  return (
    <div className="space-y-4">
      {EXPERIENCE.map((job, i) => (
        <Card key={i} className="p-8 group hover:bg-white/5 transition-colors">
          <div className="flex flex-col md:flex-row md:items-baseline justify-between mb-4">
            <h3 className="text-xl font-medium text-slate-200 group-hover:text-cyan-400 transition-colors">
              {job.company}
            </h3>
            <span className="font-mono text-xs text-slate-600">{job.date}</span>
          </div>

          <div className="mb-6 font-mono text-xs text-cyan-500/70 uppercase tracking-widest">
            {job.role}
          </div>

          <div className="space-y-2">
            {job.details.map((detail, j) => (
              <p key={j} className="text-slate-400 text-sm font-light">
                — {detail}
              </p>
            ))}
          </div>
        </Card>
      ))}
    </div>
  );
}

function ProjectsView() {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      {PROJECTS.map((project, i) => (
        <Card
          key={i}
          className="p-8 flex flex-col h-64 justify-between hover:border-cyan-500/30 group"
        >
          <div>
            <div className="font-mono text-xs text-cyan-500 mb-2">{project.sub}</div>
            <h3 className="text-2xl text-slate-200 font-light mb-4 group-hover:text-white transition-colors">
              {project.title}
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">{project.desc}</p>
          </div>
          <div className="font-mono text-xs text-slate-600 pt-4 border-t border-white/5 mt-4">
            {project.tech}
          </div>
        </Card>
      ))}
    </div>
  );
}

function TeachingView() {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      {TEACHING.map((course, i) => (
        <Card
          key={i}
          className="p-8 flex flex-col h-64 justify-between hover:border-cyan-500/30 group"
        >
          <div>
            <div className="font-mono text-xs text-cyan-500 mb-2">{course.sub}</div>
            <h3 className="text-2xl text-slate-200 font-light mb-4 group-hover:text-white transition-colors">
              {course.title}
            </h3>
            <p className="text-sm text-slate-400 leading-relaxed">{course.desc}</p>
          </div>
        </Card>
      ))}
    </div>
  );
}

function ResumeView() {
  return (
    <div className="flex items-center justify-center h-[50vh]">
      <Card className="p-12 text-center hover:border-cyan-500/50 group cursor-pointer">
        <Download
          className="mx-auto mb-6 text-slate-600 group-hover:text-cyan-400 transition-colors"
          size={32}
        />
        <h3 className="text-xl text-slate-200 font-light mb-2">GJ_RESUME_2025.PDF</h3>
        <p className="font-mono text-xs text-slate-500 uppercase tracking-widest mb-8">
          Secure Document // 4.2 MB
        </p>
        <span className="px-6 py-3 bg-white/5 text-xs font-mono text-cyan-400 uppercase tracking-widest rounded-sm group-hover:bg-cyan-500/10 transition-colors">
          Initiate Download
        </span>
      </Card>
    </div>
  );
}

// MAIN APP
const SECTIONS = [
  { key: "about",      label: "About Me"  },
  { key: "experience", label: "My Work"   },
  { key: "projects",   label: "Projects"  },
  { key: "teaching",   label: "Teaching"  },
  { key: "resume",     label: "Contact"   },
];

export default function Portfolio() {
  const canvasRef = useFluidCanvas();
  const [view, setView] = useState("menu");

  const SECTION_COMPONENTS = {
    about:      <AboutView />,
    experience: <ExperienceView />,
    projects:   <ProjectsView />,
    teaching:   <TeachingView />,
    resume:     <ResumeView />,
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-slate-950 text-slate-200 font-sans selection:bg-cyan-500/30">
      {/* Background fluid simulation canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />

      {/* Foreground UI */}
      <div className="relative z-10 w-full h-full">

        {/* ── MENU ── */}
        {view === "menu" && (
          <div className="flex flex-col items-start justify-center h-full max-w-5xl w-full mx-auto px-8 md:px-16">

            {/* Hero name */}
            <div className="mb-24 mix-blend-overlay">
              <h1 className="text-7xl md:text-9xl font-bold text-white tracking-tighter opacity-80">
                GABE
              </h1>
              <h1 className="text-7xl md:text-9xl font-bold text-white/50 tracking-tighter -mt-4 md:-mt-8">
                JOHNSON
              </h1>
            </div>

            {/* Navigation */}
            <nav className="w-full max-w-sm border-t border-white/10 pt-4">
              {SECTIONS.map(({ key, label }) => (
                <NavItem
                  key={key}
                  label={label}
                  onClick={() => setView(key)}
                  active={view === key}
                />
              ))}
            </nav>

            {/* Social links */}
            <div className="fixed bottom-12 right-12 flex gap-8 text-slate-600">
              <a href="https://github.com/gabejohnsnn" className="hover:text-cyan-400 transition-colors">
                <Github size={20} />
              </a>
              <a href="https://substack.com/gabejohnsnn" className="hover:text-cyan-400 transition-colors">
                <Bookmark size={20} />
              </a>
              <a href="https://www.linkedin.com/in/gabojohnson/" className="hover:text-cyan-400 transition-colors">
                <Linkedin size={20} />
              </a>
              <a href="mailto:gabriel.johnson@uvm.edu" className="hover:text-cyan-400 transition-colors">
                <Mail size={20} />
              </a>
            </div>
          </div>
        )}

        {/* ── SECTION VIEWS ── */}
        {view !== "menu" && (
          <div className="h-full w-full overflow-y-auto px-6 py-12 md:p-24">
            <div className="max-w-4xl mx-auto">
              <BackButton onClick={() => setView("menu")} />
              {SECTION_COMPONENTS[view]}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
