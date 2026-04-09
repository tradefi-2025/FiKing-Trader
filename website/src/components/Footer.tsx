import Link from "next/link";
import Image from "next/image";
import { Mail, ExternalLink } from "lucide-react";

const footerLinks = {
  Platform: [
    { label: "How It Works", href: "/how-it-works" },
    { label: "Use Cases", href: "/use-cases" },
    { label: "Research", href: "/research" },
  ],
  Company: [
    { label: "About Us", href: "/about" },
    { label: "Team", href: "/about#team" },
    { label: "Contact", href: "/about#contact" },
  ],
  Resources: [
    { label: "Documentation", href: "#" },
    { label: "API Reference", href: "#" },
    { label: "Blog", href: "#" },
  ],
};

export default function Footer() {
  return (
    <footer className="border-t border-border bg-surface">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="py-12 grid grid-cols-2 md:grid-cols-5 gap-8">
          {/* Brand */}
          <div className="col-span-2">
            <Link href="/" className="flex items-center gap-2.5 mb-4">
              <Image
                src="/Logo.png"
                alt="Stratify"
                width={32}
                height={32}
              />
              <span className="text-lg font-bold text-white">Stratify</span>
            </Link>
            <p className="text-sm text-muted max-w-xs leading-relaxed">
              Trading intelligence from multimodal understanding. We blend years
              of market data with news signals through large AI models.
            </p>
            <div className="flex gap-3 mt-5">
              {[
                { icon: Mail, href: "mailto:admin@stratify.finance" },
                { icon: ExternalLink, href: "#" },
                { icon: ExternalLink, href: "#" },
              ].map(({ icon: Icon, href }, i) => (
                <a
                  key={i}
                  href={href}
                  className="flex h-9 w-9 items-center justify-center rounded-lg border border-border text-muted hover:text-white hover:border-primary/50 transition-colors"
                >
                  <Icon className="h-4 w-4" />
                </a>
              ))}
            </div>
          </div>

          {/* Link columns */}
          {Object.entries(footerLinks).map(([title, items]) => (
            <div key={title}>
              <h4 className="text-sm font-semibold text-white mb-4">{title}</h4>
              <ul className="space-y-2.5">
                {items.map((item) => (
                  <li key={item.label}>
                    <Link
                      href={item.href}
                      className="text-sm text-muted hover:text-white transition-colors"
                    >
                      {item.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Bottom bar */}
        <div className="border-t border-border py-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted">
            &copy; {new Date().getFullYear()} Stratify. All rights reserved.
          </p>
          <div className="flex gap-6">
            <Link href="#" className="text-xs text-muted hover:text-white transition-colors">
              Privacy Policy
            </Link>
            <Link href="#" className="text-xs text-muted hover:text-white transition-colors">
              Terms of Service
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
