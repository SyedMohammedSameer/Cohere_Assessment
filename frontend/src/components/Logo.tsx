/**
 * Brand mark and small UI icons.
 *
 * `CohereMark` renders the official Cohere logo from `public/logo.svg`, paired
 * with the lowercase "cohere" wordmark in the header.
 */

export function CohereMark({ className = "" }: { className?: string }) {
  return <img src="/logo.svg" alt="Cohere" className={className} draggable={false} />;
}

export function MenuIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" aria-hidden="true">
      <path
        d="M4 6h16M4 12h16M4 18h16"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function CloseIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none" aria-hidden="true">
      <path
        d="M6 6l12 12M18 6L6 18"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}
