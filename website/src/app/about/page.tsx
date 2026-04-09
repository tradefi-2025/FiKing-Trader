"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import AnimatedSection from "@/components/AnimatedSection";
import {
  Target,
  Eye,
  Heart,
  Globe,
  GraduationCap,
  Lightbulb,
  ArrowRight,
  MapPin,
  Mail,
} from "lucide-react";

const values = [
  {
    icon: Lightbulb,
    title: "Innovation First",
    desc: "We push the boundaries of what AI can achieve in financial markets, pursuing novel multimodal approaches.",
  },
  {
    icon: Eye,
    title: "Transparency",
    desc: "Our models are explainable and our methodology is open. Clients understand how decisions are made.",
  },
  {
    icon: Heart,
    title: "Integrity",
    desc: "We build trust through rigorous risk management and honest performance reporting.",
  },
  {
    icon: GraduationCap,
    title: "Research-Driven",
    desc: "Every feature is grounded in academic research and validated through extensive backtesting.",
  },
];

const team = [
  {
    initials: "KF",
    name: "Kian Feizabadi",
    role: "Chief Executive Officer",
    location: "Grenoble",
    color: "from-blue-500 to-blue-600",
  },
  {
    initials: "AB",
    name: "Ali Borjoueizadeh",
    role: "Product Lead",
    location: "Grenoble",
    color: "from-purple-500 to-purple-600",
  },
  {
    initials: "AS",
    name: "Amin Saeidi Kelishami",
    role: "AI Lead",
    location: "Grenoble",
    color: "from-teal-500 to-teal-600",
  },
  {
    initials: "HA",
    name: "Hossein Abdolmotallebi",
    role: "Operations & Backend Lead",
    location: "Paris",
    color: "from-orange-500 to-orange-600",
  },
  {
    initials: "MS",
    name: "Mahdi Sojoudi",
    role: "Finance R&D Lead",
    location: "Grenoble",
    color: "from-pink-500 to-pink-600",
  },
  {
    initials: "RD",
    name: "Reza Dolati",
    role: "Data & Infrastructure Lead",
    location: "Grenoble",
    color: "from-indigo-500 to-indigo-600",
  },
  {
    initials: "AH",
    name: "Arman Hosseini",
    role: "AI Researcher",
    location: "Virginia",
    color: "from-cyan-500 to-cyan-600",
  },
];

const milestones = [
  { year: "2024", event: "Research began on multimodal trading models at Grenoble" },
  { year: "2025", event: "Core team assembled — seven co-founders from AI, Finance & Engineering" },
  { year: "2025", event: "First prototype: Kronos + FLANG-BERT pipeline processing live data" },
  { year: "2026", event: "Stratify platform enters private beta with institutional partners" },
];

export default function AboutPage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-40" />
        <div className="absolute top-1/3 left-1/4 w-[600px] h-[600px] rounded-full bg-accent/10 blur-[120px]" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 pt-24 pb-16 lg:pt-32 lg:pb-24">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="max-w-3xl"
          >
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium border border-accent/30 text-accent bg-accent/5 mb-6">
              <Globe className="h-3 w-3" />
              About Stratify
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.1]">
              Building the Future of{" "}
              <span className="gradient-text">Systematic Trading</span>
            </h1>
            <p className="mt-6 text-lg text-muted max-w-2xl leading-relaxed">
              We are a team of seven co-founders from the intersection of
              artificial intelligence, quantitative finance, and software
              engineering — united by the belief that multimodal understanding
              is the next frontier of trading intelligence.
            </p>
          </motion.div>
        </div>
      </section>

      {/* Mission / Vision */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-8">
            <AnimatedSection>
              <div className="glass-light rounded-2xl p-8 h-full">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-primary to-blue-600 mb-5">
                  <Target className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Our Mission</h3>
                <p className="text-muted leading-relaxed">
                  To democratize AI-powered trading by giving every investor
                  access to institutional-grade multimodal intelligence. We
                  believe that blending time-series analysis with contextual
                  news understanding creates a fundamentally superior approach
                  to market analysis.
                </p>
              </div>
            </AnimatedSection>
            <AnimatedSection delay={0.15}>
              <div className="glass-light rounded-2xl p-8 h-full">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-accent to-purple-600 mb-5">
                  <Eye className="h-6 w-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">Our Vision</h3>
                <p className="text-muted leading-relaxed">
                  A world where intelligent, AI-driven trading strategies are
                  accessible to everyone — not just hedge funds and large
                  institutions. We envision a platform where users create
                  custom trading agents without writing a single line of code,
                  powered by cutting-edge multimodal models.
                </p>
              </div>
            </AnimatedSection>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              What We <span className="gradient-text">Stand For</span>
            </h2>
          </AnimatedSection>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {values.map((v, i) => (
              <AnimatedSection key={v.title} delay={i * 0.1}>
                <div className="glass-light rounded-2xl p-6 h-full text-center hover:border-primary/30 transition-colors">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 mx-auto mb-4">
                    <v.icon className="h-5 w-5 text-primary-light" />
                  </div>
                  <h4 className="text-base font-semibold text-white mb-2">{v.title}</h4>
                  <p className="text-sm text-muted leading-relaxed">{v.desc}</p>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Our <span className="gradient-text">Journey</span>
            </h2>
          </AnimatedSection>
          <div className="max-w-2xl mx-auto">
            {milestones.map((m, i) => (
              <AnimatedSection key={i} delay={i * 0.1}>
                <div className="flex gap-6 mb-8 last:mb-0">
                  <div className="flex flex-col items-center">
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-xs font-bold text-white flex-shrink-0">
                      {m.year.slice(-2)}
                    </div>
                    {i < milestones.length - 1 && (
                      <div className="w-px h-full bg-border mt-2" />
                    )}
                  </div>
                  <div className="pb-8">
                    <span className="text-xs font-mono text-primary-light">{m.year}</span>
                    <p className="text-sm text-muted mt-1 leading-relaxed">{m.event}</p>
                  </div>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section id="team" className="py-20 relative">
        <div className="absolute inset-0 bg-gradient-to-b from-background via-surface to-background" />
        <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection className="text-center mb-14">
            <h2 className="text-3xl sm:text-4xl font-bold text-white">
              Meet the <span className="gradient-text">Team</span>
            </h2>
            <p className="mt-4 text-muted max-w-xl mx-auto">
              Seven co-founders building the future of systematic trading.
            </p>
          </AnimatedSection>

          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {team.map((t, i) => (
              <AnimatedSection key={t.name} delay={i * 0.08}>
                <div className="glass-light rounded-2xl p-6 text-center hover:border-primary/30 transition-all group">
                  <div
                    className={`flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br ${t.color} mx-auto mb-4 text-xl font-bold text-white`}
                  >
                    {t.initials}
                  </div>
                  <h4 className="text-base font-semibold text-white group-hover:text-primary-light transition-colors">
                    {t.name}
                  </h4>
                  <p className="text-xs font-medium text-primary-light uppercase tracking-wider mt-1">
                    {t.role}
                  </p>
                  <div className="flex items-center justify-center gap-1 mt-3 text-xs text-muted">
                    <MapPin className="h-3 w-3" />
                    {t.location}
                  </div>
                </div>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* Contact */}
      <section id="contact" className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <AnimatedSection>
            <div className="relative rounded-3xl overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-accent/15 via-primary/10 to-teal/10" />
              <div className="relative glass-light rounded-3xl px-8 py-16 sm:px-16 text-center">
                <Mail className="h-10 w-10 text-primary-light mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-white mb-3">
                  Get In Touch
                </h2>
                <p className="text-muted max-w-lg mx-auto mb-8">
                  Interested in Stratify? Whether you&apos;re an investor, researcher,
                  or potential partner — we&apos;d love to hear from you.
                </p>
                <Link
                  href="mailto:admin@stratify.finance"
                  className="group inline-flex items-center gap-2 px-8 py-3.5 text-sm font-semibold text-white rounded-xl bg-gradient-to-r from-primary to-accent hover:opacity-90 transition-opacity"
                >
                  admin@stratify.finance
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-0.5 transition-transform" />
                </Link>
              </div>
            </div>
          </AnimatedSection>
        </div>
      </section>
    </>
  );
}
