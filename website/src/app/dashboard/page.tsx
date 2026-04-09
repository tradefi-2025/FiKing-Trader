"use client";

import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  TrendingUp,
  Activity,
  Clock,
  Plus,
  ArrowUpRight,
  ArrowDownRight,
  Bot,
} from "lucide-react";

const stats = [
  { label: "Active Agents", value: "3", icon: Bot, change: "+1 this week" },
  { label: "Total Signals", value: "1,247", icon: Activity, change: "+86 today" },
  { label: "Win Rate", value: "64.2%", icon: TrendingUp, change: "+2.1% vs last month" },
  { label: "Avg. Response", value: "0.8s", icon: Clock, change: "Healthy" },
];

const agents = [
  {
    name: "AAPL Momentum Agent",
    equity: "AAPL",
    status: "running",
    signals: 412,
    winRate: 67.3,
    lastSignal: "BUY",
    lastSignalTime: "2 min ago",
    pnl: "+4.2%",
    pnlPositive: true,
  },
  {
    name: "TSLA Volatility Agent",
    equity: "TSLA",
    status: "running",
    signals: 389,
    winRate: 59.1,
    lastSignal: "SELL",
    lastSignalTime: "5 min ago",
    pnl: "+1.8%",
    pnlPositive: true,
  },
  {
    name: "MSFT News Sentiment",
    equity: "MSFT",
    status: "running",
    signals: 446,
    winRate: 66.4,
    lastSignal: "HOLD",
    lastSignalTime: "1 min ago",
    pnl: "+3.1%",
    pnlPositive: true,
  },
];

export default function DashboardPage() {
  return (
    <section className="min-h-[calc(100vh-4rem)] relative">
      <div className="absolute inset-0 bg-grid opacity-20" />

      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8"
        >
          <div>
            <h1 className="text-2xl font-bold text-white">Dashboard</h1>
            <p className="text-sm text-muted mt-1">
              Overview of your trading agents and performance.
            </p>
          </div>
          <Link
            href="#"
            className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
          >
            <Plus className="h-4 w-4" />
            Create Agent
          </Link>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"
        >
          {stats.map((s) => (
            <div
              key={s.label}
              className="glass-light rounded-2xl p-5 hover:border-primary/30 transition-colors"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-medium text-muted uppercase tracking-wider">
                  {s.label}
                </span>
                <s.icon className="h-4 w-4 text-primary-light" />
              </div>
              <p className="text-2xl font-bold text-white">{s.value}</p>
              <p className="text-xs text-muted mt-1">{s.change}</p>
            </div>
          ))}
        </motion.div>

        {/* Agents Table */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          className="glass-light rounded-2xl overflow-hidden"
        >
          <div className="px-6 py-4 border-b border-border flex items-center justify-between">
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <Bot className="h-4 w-4 text-primary-light" />
              Active Agents
            </h2>
            <span className="text-xs text-muted">
              {agents.length} agents running
            </span>
          </div>

          {/* Desktop table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  {["Agent", "Equity", "Status", "Signals", "Win Rate", "Last Signal", "PnL"].map(
                    (h) => (
                      <th
                        key={h}
                        className="px-6 py-3 text-left text-[10px] font-semibold text-muted uppercase tracking-wider"
                      >
                        {h}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {agents.map((a) => (
                  <tr
                    key={a.name}
                    className="border-b border-border/50 hover:bg-white/[.02] transition-colors"
                  >
                    <td className="px-6 py-4">
                      <p className="text-sm font-medium text-white">{a.name}</p>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-xs font-mono text-primary-light bg-primary/10 px-2 py-0.5 rounded-md">
                        {a.equity}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center gap-1.5 text-xs">
                        <span className="h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
                        <span className="text-green-400 capitalize">{a.status}</span>
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-muted">{a.signals.toLocaleString()}</td>
                    <td className="px-6 py-4 text-sm text-white">{a.winRate}%</td>
                    <td className="px-6 py-4">
                      <div>
                        <span
                          className={`text-xs font-semibold px-2 py-0.5 rounded-md ${
                            a.lastSignal === "BUY"
                              ? "bg-green-500/10 text-green-400"
                              : a.lastSignal === "SELL"
                              ? "bg-red-500/10 text-red-400"
                              : "bg-yellow-500/10 text-yellow-400"
                          }`}
                        >
                          {a.lastSignal}
                        </span>
                        <p className="text-[10px] text-muted mt-0.5">{a.lastSignalTime}</p>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="flex items-center gap-1 text-sm font-medium text-green-400">
                        {a.pnlPositive ? (
                          <ArrowUpRight className="h-3.5 w-3.5" />
                        ) : (
                          <ArrowDownRight className="h-3.5 w-3.5" />
                        )}
                        {a.pnl}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile cards */}
          <div className="md:hidden divide-y divide-border/50">
            {agents.map((a) => (
              <div key={a.name} className="p-5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <p className="text-sm font-medium text-white">{a.name}</p>
                    <span className="text-xs font-mono text-primary-light">{a.equity}</span>
                  </div>
                  <span className="flex items-center gap-1 text-sm font-medium text-green-400">
                    <ArrowUpRight className="h-3.5 w-3.5" />
                    {a.pnl}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <p className="text-[10px] text-muted uppercase">Status</p>
                    <span className="inline-flex items-center gap-1 text-xs text-green-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
                      {a.status}
                    </span>
                  </div>
                  <div>
                    <p className="text-[10px] text-muted uppercase">Win Rate</p>
                    <p className="text-xs text-white">{a.winRate}%</p>
                  </div>
                  <div>
                    <p className="text-[10px] text-muted uppercase">Last Signal</p>
                    <span
                      className={`text-xs font-semibold ${
                        a.lastSignal === "BUY"
                          ? "text-green-400"
                          : a.lastSignal === "SELL"
                          ? "text-red-400"
                          : "text-yellow-400"
                      }`}
                    >
                      {a.lastSignal}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.3 }}
          className="mt-6 glass rounded-2xl px-6 py-4 flex items-center gap-3"
        >
          <Image src="/Logo.png" alt="Stratify" width={20} height={20} className="flex-shrink-0" />
          <p className="text-xs text-muted">
            This is a preview dashboard with sample data. Full dashboard
            functionality — including live agent management, real-time signals,
            and portfolio analytics — will be available when the platform
            launches.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
