document.addEventListener('DOMContentLoaded', () => {
    // --- Loading Animation ---
    const loadingScreen = document.querySelector('.loading');
    window.addEventListener('load', () => {
        loadingScreen.classList.add('fade-out');
    });

    // --- Scroll Reveal Animation ---
    const scrollElements = document.querySelectorAll('.scroll-reveal');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
            }
        });
    }, { threshold: 0.1 });

    scrollElements.forEach(el => observer.observe(el));
    
    // --- Smooth scroll for anchor links ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
