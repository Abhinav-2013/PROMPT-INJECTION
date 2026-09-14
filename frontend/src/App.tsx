import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  FileSearch,
  History,
  Menu,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import {
  BrowserRouter,
  NavLink,
  Route,
  Routes,
} from 'react-router-dom'

import Dashboard from './pages/Dashboard'
import Analyze from './pages/Analyze'
import Documents from './pages/Documents'
import Explainability from './pages/Explainability'
import HistoryPage from './pages/History'
import SystemStatus from './pages/SystemStatus'

function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <BrowserRouter>
      <main className="relative min-h-screen overflow-hidden bg-[#f5f7fa] text-[#111318]">

        {/* Animated background */}
        <div className="pointer-events-none fixed inset-0 overflow-hidden">

          <motion.div
            className="absolute -left-32 -top-32 h-96 w-96 rounded-full bg-blue-200/40 blur-3xl"
            animate={{
              x: [0, 80, 0],
              y: [0, 50, 0],
            }}
            transition={{
              duration: 12,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          <motion.div
            className="absolute -right-40 top-1/4 h-[30rem] w-[30rem] rounded-full bg-violet-200/30 blur-3xl"
            animate={{
              x: [0, -70, 0],
              y: [0, 80, 0],
            }}
            transition={{
              duration: 15,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

          <motion.div
            className="absolute bottom-[-10rem] left-1/3 h-[28rem] w-[28rem] rounded-full bg-cyan-100/50 blur-3xl"
            animate={{
              x: [0, 60, 0],
            }}
            transition={{
              duration: 14,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
          />

        </div>

        {/* Navigation */}
        <nav className="relative z-50 mx-auto max-w-7xl px-6 py-5 lg:px-8">

          <div className="flex items-center justify-between rounded-[1.5rem] border border-white/70 bg-white/45 px-4 py-3 shadow-lg shadow-slate-900/5 backdrop-blur-2xl">

            <NavLink
              to="/"
              className="flex items-center gap-3"
              onClick={() => setMobileMenuOpen(false)}
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/80 bg-white/60 shadow-sm">
                <ShieldCheck className="h-5 w-5 text-slate-800" />
              </div>

              <div>
                <p className="text-sm font-semibold tracking-tight">
                  Prompt Injection Detection System
                </p>
                <p className="hidden text-[11px] text-slate-400 sm:block">
                  Intelligent threat detection
                </p>
              </div>
            </NavLink>

            {/* Desktop navigation */}
            <div className="hidden items-center gap-1 rounded-full border border-white/70 bg-white/35 p-1 backdrop-blur-xl md:flex">

              <NavigationItem to="/" label="Dashboard" />
              <NavigationItem to="/analyze" label="Analyze" />
              <NavigationItem to="/documents" label="Documents" />
              <NavigationItem to="/explainability" label="Explainability" />
              <NavigationItem to="/history" label="History" />

            </div>

            <div className="flex items-center gap-2">

              {/* Status */}
              <NavLink
                to="/status"
                className="hidden items-center gap-2 rounded-full border border-white/70 bg-white/45 px-3 py-2 backdrop-blur-xl sm:flex"
              >
                <span className="relative flex h-2.5 w-2.5">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-slate-400 opacity-40" />
                  <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-slate-400" />
                </span>

                <span className="text-xs font-medium text-slate-500">
                  System Status
                </span>
              </NavLink>

              {/* Mobile menu */}
              <button
                type="button"
                aria-label="Open navigation menu"
                onClick={() => setMobileMenuOpen((open) => !open)}
                className="flex h-10 w-10 items-center justify-center rounded-full border border-white/70 bg-white/50 md:hidden"
              >
                {mobileMenuOpen ? (
                  <X className="h-5 w-5" />
                ) : (
                  <Menu className="h-5 w-5" />
                )}
              </button>

            </div>

          </div>

          {/* Mobile navigation */}
          <AnimatePresence>
            {mobileMenuOpen && (
              <motion.div
                initial={{ opacity: 0, y: -10, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.98 }}
                className="mt-3 rounded-[1.5rem] border border-white/70 bg-white/55 p-3 shadow-xl backdrop-blur-2xl md:hidden"
              >
                <MobileNavigation
                  closeMenu={() => setMobileMenuOpen(false)}
                />
              </motion.div>
            )}
          </AnimatePresence>

        </nav>

        {/* Page content */}
        <div className="relative z-10">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/status" element={<SystemStatus />} />
          </Routes>
        </div>

      </main>
    </BrowserRouter>
  )
}

function NavigationItem({
  to,
  label,
}: {
  to: string
  label: string
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `rounded-full px-4 py-2 text-xs font-medium transition ${
          isActive
            ? 'bg-white/80 text-slate-900 shadow-sm'
            : 'text-slate-500 hover:bg-white/50 hover:text-slate-800'
        }`
      }
    >
      {label}
    </NavLink>
  )
}

function MobileNavigation({
  closeMenu,
}: {
  closeMenu: () => void
}) {
  const items = [
    {
      to: '/',
      label: 'Dashboard',
      icon: <Activity className="h-4 w-4" />,
    },
    {
      to: '/analyze',
      label: 'Analyze',
      icon: <Sparkles className="h-4 w-4" />,
    },
    {
      to: '/documents',
      label: 'Documents',
      icon: <FileSearch className="h-4 w-4" />,
    },
    {
      to: '/explainability',
      label: 'Explainability',
      icon: <Sparkles className="h-4 w-4" />,
    },
    {
      to: '/history',
      label: 'History',
      icon: <History className="h-4 w-4" />,
    },
    {
      to: '/status',
      label: 'System Status',
      icon: <ShieldCheck className="h-4 w-4" />,
    },
  ]

  return (
    <div className="space-y-1">
      {items.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          onClick={closeMenu}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition ${
              isActive
                ? 'bg-white/80 text-slate-900'
                : 'text-slate-500 hover:bg-white/50'
            }`
          }
        >
          {item.icon}
          {item.label}
        </NavLink>
      ))}
    </div>
  )
}

export default App